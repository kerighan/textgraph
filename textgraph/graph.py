import numpy as np


class TextGraph:
    def __init__(
        self,
        min_tf=2,
        min_df=1,
        max_df=.5,
        sim_threshold=.5,
        pmi_threshold=.1,
        wv_threshold=.5,
        max_features=None,
        window=10,
        stopwords=["fr", "en"],
        freq="1D"
    ):
        self.pmi_threshold = pmi_threshold
        self.sim_threshold = sim_threshold
        self.wv_threshold = wv_threshold
        self.min_tf = min_tf
        self.min_df = min_df
        self.max_df = max_df
        self.window = window
        self.stopwords = stopwords
        self.freq = freq
        self.max_features = max_features

    def fit(self, documents, dates=None, wv=None):
        if wv is None:
            return self._fit_withouth_wv(documents, dates=dates)
        else:
            return self._fit_with_wv(documents, wv, dates=dates)

    def _fit_withouth_wv(self, documents, dates=None):
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
        self.texts = documents
        self.dates = dates

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
        self.G = G
        return G

    def _fit_with_wv(self, documents, wv, dates=None):
        import math

        import networkx as nx
        import numpy as np
        from convectors import load_model
        from convectors.layers import (Contract, Lemmatize, Phrase, TfIdf,
                                       Tokenize)
        from convectors.linguistics import cooc
        from scipy.sparse import find

        nlp = Contract()
        nlp += Tokenize(stopwords=self.stopwords)
        nlp += Lemmatize()
        # nlp += Phrase()
        nlp.verbose = False
        tokens = nlp(documents)
        self.texts = documents
        self.dates = dates

        _cooc = cooc(tokens,
                     window_size=self.window,
                     undirected=True)

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
        for (word_a, word_b), o in _cooc.items():
            if word_a not in wv or word_b not in wv:
                continue
            # weight = float(weight)
            weight = math.log(o+1) * (wv[word_a] @ wv[word_b])
            if weight > self.wv_threshold:
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
        self.G = G
        return G

    def topics(self, resolution=1):
        import networkx as nx
        try:
            from cylouvain import best_partition, modularity
        except:
            from community import best_partition, modularity
        from convectors.graph import relabel

        community = relabel(best_partition(
            self.G, random_state=0, resolution=resolution))
        print(f"Q={modularity(community, self.G):.2f}")

        cm2nodes = {}
        for node, cm in community.items():
            cm2nodes.setdefault(cm, []).append(node)

        topics = []
        for i in range(len(cm2nodes)):
            nodes = cm2nodes[i]
            H = nx.subgraph(self.G, nodes)
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

            # if self.texts is not None:
            #     docs = self.texts.iloc[docs]

            topics.append({
                "docs": docs,
                "keywords": keywords,
                "size": len(docs)})
        topics = sorted(topics, key=lambda x: x["size"], reverse=True)
        self.topics = topics
        return topics

    def setup_sentiment(self):
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer, pipeline)
        tokenizer = AutoTokenizer.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment")
        model = AutoModelForSequenceClassification.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment")
        self.classifier = pipeline("sentiment-analysis",
                                   model=model, tokenizer=tokenizer)

    def get_sentiment(self, texts):
        labels = [int(it["label"][0]) for it in self.classifier(texts)]
        mean = np.mean(labels)
        sentiment = np.bincount(labels, minlength=6)[1:]
        return sentiment, mean

    def dashboard(self, texts=None, name="topic", topn=20, sentiment=True):
        import itertools
        from collections import Counter

        import pandas as pd
        from convectors.layers import NER
        from conviction import Board, Tab
        from conviction.components import Bar, IndicatorCards, Table
        from tqdm import tqdm

        if sentiment:
            self.setup_sentiment()

        if texts is None:
            texts = self.texts

        board = Board("Topic")

        ner = NER(verbose=False)
        total = sum(t["size"] for t in self.topics)
        for i in tqdm(range(topn)):
            tab = Tab(title=str(i))

            topic = self.topics[i]
            size = topic["size"]
            rel_size = 100*size/total

            docs = texts.iloc[topic["docs"]]
            t = pd.DataFrame()
            t["text"] = docs

            indicators = [
                {"indicator": int(size), "title": "size",
                 "description": "nombre de documents"},
                {"indicator": f"{rel_size:.2f}",
                 "title": "pourcentage", "description": "pourcentage du total"}
            ]

            if sentiment:
                sent, mean_sent = self.get_sentiment(docs.tolist()[:2000])
                bar = Bar(title="Sentiment", labels=range(len(sent)))
                bar.add_dataset(
                    [int(a) for a in sent],
                    label="sentiment",
                    background_color="rgba(33, 150, 243, 0.2)",
                    border_width=2,
                    border_color="rgb(33, 150, 243)"
                )
                tab.add_component(bar)

            if self.dates is not None:
                from conviction.components import Line
                from virality.sigmoid import fit_sigmoid
                dates = pd.to_datetime(
                    self.dates.iloc[topic["docs"]]).apply(
                        lambda x: x.replace(tzinfo=None))
                dates = pd.DataFrame(dates)
                dates.columns = ["created_time"]
                dates = pd.DataFrame(dates.groupby(
                    pd.Grouper(key="created_time", freq="1D")).size())
                dates.reset_index(inplace=True)
                dates.columns = ["date", "volume"]

                # sir = SIR(dates.volume.values)["R_0"]
                _, sir = fit_sigmoid(dates.volume.values)
                indicators.append(
                    {"indicator": f"{sir:.2f}", "title": "R0", "description": "virality"})

                cards = IndicatorCards(
                    title="cards", data=pd.DataFrame(indicators),
                    style="third")
                tab.add_component(cards)

                line = Line(title="Volume",
                            labels=dates.date.apply(
                                lambda x: x.strftime("%Y-%m-%d")).tolist(),
                            description="topic volume")
                line.add_dataset(dates.volume.astype(int).tolist(),
                                 label="volume",
                                 border_width=2,
                                 background_color="rgba(33, 150, 243, 0.2)", border_color="rgb(33, 150, 243)")
                tab.add_component(line)
            else:
                cards = IndicatorCards(
                    title="cards", data=pd.DataFrame(indicators), style="half")
                tab.add_component(cards)

            table = Table(
                title="docs",
                data=t.head(50), style="highlight")
            tab.add_component(table)

            keywords = pd.DataFrame(topic["keywords"], columns=[
                                    "keyword", "centrality"]).head(25)
            kw = Table(
                title="docs", data=keywords, style="highlight")
            tab.add_component(kw)

            res = ner(docs)
            count_per = Counter(itertools.chain(
                *res.apply(lambda x: [y.lower() for y, c in x if c == "PER"])))
            # print(count_per.most_common(20))
            count_org = Counter(itertools.chain(
                *res.apply(lambda x: [y.lower() for y, c in x if c == "ORG"])))
            # print(count_org.most_common(20))
            count_loc = Counter(itertools.chain(
                *res.apply(lambda x: [y.lower() for y, c in x if c == "LOC"])))
            # print(count_loc.most_common(20))

            # texts = ". ".join(docs.head(20).tolist())
            # print(summarizer([texts[:1024]])[0][0]["summary_text"])

            # print(f"size={size}")
            # print(f"relative_size={100*size/total:.2f}%")
            # print()
            # print()
            board.add_tab(tab)
        board.render_to(name)

    def show(self, topn=20):
        import itertools
        from collections import Counter

        from convectors.huggingface import Summarize
        from convectors.layers import NER

        summarizer = Summarize(verbose=False)
        ner = NER(verbose=False)
        total = sum(t["size"] for t in self.topics)
        for i in range(topn):
            topic = self.topics[i]
            size = topic["size"]
            print(i + 1)
            docs = self.texts.iloc[topic["docs"]]
            print(docs.head(20))
            print(topic["keywords"][:50])

            res = ner(docs)
            count_per = Counter(itertools.chain(
                *res.apply(lambda x: [y.lower() for y, c in x if c == "PER"])))
            print(count_per.most_common(20))
            count_org = Counter(itertools.chain(
                *res.apply(lambda x: [y.lower() for y, c in x if c == "ORG"])))
            print(count_org.most_common(20))
            count_loc = Counter(itertools.chain(
                *res.apply(lambda x: [y.lower() for y, c in x if c == "LOC"])))
            print(count_loc.most_common(20))

            # texts = ". ".join(docs.head(20).tolist())
            # print(summarizer([texts[:1024]])[0][0]["summary_text"])

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
        import os

        import pandas as pd
        from flowing2.bubble import bubble

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
