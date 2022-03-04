from numpy import isin


class TextGraph:
    def __init__(
        self,
        min_tf=2,
        min_df=1,
        max_df=.5,
        sim_threshold=.5,
        pmi_threshold=.1,
        max_features=None,
        window=10,
        stopwords=["fr", "en"],
        freq="1D"
    ):
        self.pmi_threshold = pmi_threshold
        self.sim_threshold = sim_threshold
        self.min_tf = min_tf
        self.min_df = min_df
        self.max_df = max_df
        self.window = window
        self.stopwords = stopwords
        self.freq = freq
        self.max_features = max_features

    def fit_transform(self, documents, dates=None):
        import networkx as nx
        import numpy as np
        from convectors.layers import (Contract, Lemmatize, Phrase, TfIdf,
                                       Tokenize)
        from convectors.linguistics import pmi
        from scipy.sparse import find

        nlp = Contract()
        nlp += Tokenize(stopwords=self.stopwords)
        nlp += Lemmatize()
        nlp += Phrase()
        nlp.verbose = False
        tokens = nlp(documents)

        _pmi = pmi(tokens,
                   normalize=True,
                   min_count=self.min_tf,
                   window_size=self.window,
                   undirected=True,
                   minimum=self.pmi_threshold)

        vectorizer = TfIdf(
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=self.max_features,
            verbose=False)
        X = vectorizer(tokens)
        features = vectorizer.vectorizer.get_feature_names_out()

        # word-word edges
        edges = []
        vocab = set()
        for (word_a, word_b), weight in _pmi.items():
            weight = float(weight)
            edges.append((word_a, word_b, weight))
            vocab.add(word_a)
            vocab.add(word_b)

        # word-doc edges
        for i in range(X.shape[0]):
            start = X.indptr[i]
            end = X.indptr[i + 1]
            indices = X.indices[start:end]
            weights = X.data[start:end]
            for word_id, weight in zip(indices, weights):
                word = features[word_id]
                if word not in vocab:
                    continue
                weight = float(weight)
                edges.append((i, word, weight))

        if dates is not None:
            import pandas as pd
            dates = dates.reset_index(drop=True).fillna("")
            dates = pd.to_datetime(dates, utc=True).dt.strftime("%Y-%m-%d")
            date2index = {}
            for i, date in enumerate(dates):
                date2index.setdefault(date, []).append(i)

            for indices in date2index.values():
                if len(indices) <= 1:
                    continue

                vectors = X[indices]
                B = (vectors @ vectors.T).todense()
                xs, ys = np.where(B > self.sim_threshold)

                for a, b in zip(xs, ys):
                    if a >= b:
                        continue
                    weight = B[a, b]
                    edges.append((indices[a], indices[b], weight))

        G = nx.Graph()
        G.add_nodes_from(range(X.shape[0]))
        G.add_weighted_edges_from(edges)
        return G

    def topics(self, G, texts=None, resolution=1):
        import networkx as nx
        from community import best_partition, modularity
        from convectors.graph import relabel

        community = relabel(best_partition(G, random_state=0))
        print(f"Q={modularity(community, G):.2f}")

        cm2nodes = {}
        for node, cm in community.items():
            cm2nodes.setdefault(cm, []).append(node)

        topics = []
        for i in range(len(cm2nodes)):
            nodes = cm2nodes[i]
            H = nx.subgraph(G, nodes)
            if len(nodes) < 1000:
                centrality = nx.closeness_centrality(H)
            else:
                centrality = nx.pagerank(H)
            centrality = sorted(centrality.items(),
                                key=lambda x: x[1], reverse=True)

            docs = []
            keywords = []
            for node, c in centrality:
                if isinstance(node, int):
                    docs.append(node)
                else:
                    keywords.append((node, c))

            if texts is not None:
                docs = texts.iloc[docs]

            topics.append({
                "docs": docs,
                "keywords": keywords,
                "size": len(docs)})
        topics = sorted(topics, key=lambda x: x["size"], reverse=True)
        return topics

    def show(self, topics, content=None, topn=20):
        import itertools
        from collections import Counter

        from convectors.huggingface import NER, Summarize

        summarizer = Summarize(verbose=False)
        ner = NER(verbose=False)
        total = sum(t["size"] for t in topics)
        for i in range(topn):
            topic = topics[i]
            size = topic["size"]
            print(i + 1)
            print(topic["docs"].head(20))
            print(topic["keywords"][:50])

            res = ner(topic["docs"])
            count_per = Counter(itertools.chain(
                *res.apply(lambda x: [y for y, c in x if c == "PER"])))
            print(count_per.most_common(20))
            count_org = Counter(itertools.chain(
                *res.apply(lambda x: [y for y, c in x if c == "ORG"])))
            print(count_org.most_common(20))

            texts = ". ".join(topic["docs"].head(20).tolist())
            print(summarizer([texts[:1024]])[0][0]["summary_text"])

            print(f"size={size}")
            print(f"relative_size={100*size/total:.2f}%")
            print()
            print()

    def dataviz(
            self,
            topics,
            out_folder="dataviz",
            n_keywords=20,
            bg_color="#fafbfc",
            color="#d3484f",
            font_color="#ffffff",
            size=8,
            font_size=.25,
            remove_digits=True,
            blacklist={"abonné"},
            topn=20):
        from flowing2.bubble import bubble
        import pandas as pd
        import os

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        for i, topic in enumerate(topics[:topn], 1):
            df = pd.DataFrame(topic["keywords"][:n_keywords])
            if len(df) <= 2:
                continue
            df[0] = df[0].apply(lambda x: x.replace("_", " "))
            if remove_digits:
                df = df[~df[0].apply(lambda x: all(c.isdigit() for c in x))]
            df = df[~df[0].apply(lambda x: any(y in x for y in blacklist))]

            bubble(df[1], df[0], out=f"{out_folder}/{i}.svg",
                   bg_color=bg_color,
                   color=color,
                   font_color=font_color, size=size, font_size=font_size)
