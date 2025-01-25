import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell
def _():
    # Uncomment this code if you want to run the notebook on marimo cloud
    # import micropip
    # await micropip.install("Mastodon.py")
    return


@app.cell
def _():
    import marimo as mo
    import pickle
    import requests
    import time
    import functools
    import altair as alt
    from bs4 import BeautifulSoup
    from sklearn.manifold import TSNE
    import pandas as pd

    import byota
    return (
        BeautifulSoup,
        TSNE,
        alt,
        byota,
        functools,
        mo,
        pd,
        pickle,
        requests,
        time,
    )


@app.cell
def _():
    # from mastodon import Mastodon

    # # NOTE: This code only needs to be run ONCE to register your app,
    # #       keep it commented otherwise
    # Mastodon.create_app(
    #     'my_timeline_algorithm',
    #     api_base_url = 'https://fosstodon.org',
    #     to_file = 'mastimeline_clientcred.secret'
    # )
    return


@app.cell
def _(auth_form):
    auth_form
    return


@app.cell
def _(requests):
    def is_llamafile_working(llamafile_URL):
        response = requests.request(
            url=llamafile_URL,
            method="POST",
        )
        return response.status_code == 200

    # is_llamafile_working(auth_form.value["emb_llamafile_url"])
    return (is_llamafile_working,)


@app.cell
def _(auth_form, invalid_form, is_llamafile_working, mo):
    mo.stop(invalid_form(auth_form), mo.md("**Submit the form to continue.**"))

    mo.stop(
        not is_llamafile_working(auth_form.value["emb_llamafile_url"]),
        mo.md("**Cannot access llamafile embedding server.**"),
    )

    # a dictionary mapping Timeline UI checkboxes with the respective
    # strings that identify them in the Mastodon API
    timelines_dict = {
        "tl_home": "home",
        "tl_local": "local",
        "tl_public": "public",
        "tl_hashtag": "tag",
        "tl_list": "list",
    }

    timelines = []
    for k in timelines_dict.keys():
        if auth_form.value[k]:
            tl_string = timelines_dict[k]
            if tl_string in ["tag", "list"]:
                tl_string += f'/{auth_form.value[f"{k}_txt"]}'
            timelines.append(tl_string)
    return k, timelines, timelines_dict, tl_string


@app.cell
def _(mo):
    mo.md(r"""# Getting data from my Mastodon account...""")
    return


@app.cell
def _(auth_form, byota, pickle, timelines):
    offline_mode = auth_form.value["offline_mode"]
    paginated_data = {}

    if not offline_mode:
        for tl in timelines:
            paginated_data[tl] = byota.get_mastodon_data(
                auth_form.value["login"], auth_form.value["pw"], tl
            )
        with open("data_dump.pkl", "wb") as f:
            pickle.dump(paginated_data, f)

    else:
        with open("data_dump.pkl", "rb") as f:
            paginated_data = pickle.load(f)
    return f, offline_mode, paginated_data, tl


@app.cell
def _(get_compact_data, mo, offline_mode, paginated_data, pd):
    mo.stop(paginated_data is None, mo.md(f"**Issues connecting to Mastodon:**"))

    if not offline_mode:
        df = pd.DataFrame(
            # TODO: defaulting to home for now, allow for download
            #       of multiple timelines
            get_compact_data(paginated_data["home"]), columns=["id", "text"]
        )
        df.to_csv("mastodon_data.csv", index=False)
    else:
        df = pd.read_csv("mastodon_data.csv")
    return (df,)


@app.cell
def _(mo):
    mo.md("""# My timeline""")
    return


@app.cell
def _(byota, df, functools, time):
    # tt_ = time.time()
    # lf_embeddings = byota.calculate_embeddings(df["text"],functools.partial(byota.get_llamafile_embedding, llamafile_URL=auth_form.value["emb_llamafile_url"]))
    # print(time.time() - tt_)

    tt_ = time.time()
    lf_embeddings = byota.calculate_embeddings(df["text"],functools.partial(byota.get_ollama_embedding, ollama_URL="http://localhost:11434/api/embeddings", ollama_model="all-minilm"))
    print(time.time() - tt_)

    # byota.get_ollama_embedding(" ", "http://localhost:11434/api/embeddings", "all-minilm")
    # byota.get_llamafile_embedding("", llamafile_URL=auth_form.value["emb_llamafile_url"])

    # import json

    # response = requests.request(
    #             url="http://localhost:11434/api/embeddings",
    #             method="POST",
    #             data= json.dumps({
    #                 "model": "mxbai-embed-large",
    #                 "prompt": "whatever"
    #             }),
    #         )
    # response.raise_for_status()
    return lf_embeddings, tt_


@app.cell
def _():
    # for ee in lf_embeddings:
    #     print(len(ee))
    return


@app.cell
def _(TSNE, alt, df, lf_embeddings, mo, pd):
    tsne = TSNE(n_components=2, random_state=42, perplexity=3)
    projections = tsne.fit_transform(lf_embeddings)

    # df_ = pd.DataFrame(zip(projections[:,0], projections[:,1], [class_mapping[lbl] for lbl in lbls], df['1']), columns=["x","y","lbl", "text"])
    df_ = pd.DataFrame(
        zip(projections[:, 0], projections[:, 1], df["text"], df["id"]),
        columns=["x", "y", "text", "id"],
    )

    chart = (
        alt.Chart(df_, title="Timeline Visualization")
        .mark_point()
        .encode(
            x="x",
            y="y",
            # color="lbl"
        )
    )

    chart = mo.ui.altair_chart(chart)
    return chart, df_, projections, tsne


@app.cell
def _(chart, mo):
    mo.vstack(
        [
            chart,
            chart.value[["id", "text"]] if len(chart.value) > 0 else chart.value,
        ]
    )
    return


@app.cell
def _(mo):
    query = mo.ui.text(
        value="42",
        label="Enter a post id or some free-form text to find the most similar posts:\n",
        full_width=True,
    )
    return (query,)


@app.cell
def _(query):
    query
    return


@app.cell
def _(auth_form, byota, df, lf_embeddings, query):
    byota.most_similar_to(query.value, df["text"], lf_embeddings, auth_form.value['emb_llamafile_url'])
    return


@app.cell
def _(BeautifulSoup):
    def get_compact_data(paginated_data: list) -> list[tuple[int, str]]:
        compact_data = []
        for page in paginated_data:
            for toot in page:
                id = toot.id
                cont = toot.content
                if toot.reblog:
                    id = toot.reblog.id
                    cont = toot.reblog.content
                soup = BeautifulSoup(cont, features="html.parser")
                # print(f"{id}: {soup.get_text()}")
                compact_data.append((id, soup.get_text()))
        return compact_data
    return (get_compact_data,)


@app.cell
def _(mo):
    # Create a form with multiple elements
    auth_form = (
        mo.md(
            """
        # Configuration
        **Mastodon Credentials**

        {login}         {pw}

        **Timelines**

        {tl_home} {tl_local} {tl_public}

        {tl_hashtag} {tl_hashtag_txt} {tl_list} {tl_list_txt}

        **Embeddings**

        {emb_llamafile_url}

        **Caching**

        {offline_mode}
    """
        )
        .batch(
            login=mo.ui.text(label="Login:"),
            pw=mo.ui.text(label="Password:", kind="password"),
            tl_home=mo.ui.checkbox(label="Home", value=True),
            tl_local=mo.ui.checkbox(label="Local"),
            tl_public=mo.ui.checkbox(label="Public"),
            tl_hashtag=mo.ui.checkbox(label="Hashtag"),
            tl_list=mo.ui.checkbox(label="List"),
            tl_hashtag_txt=mo.ui.text(),
            tl_list_txt=mo.ui.text(),
            emb_llamafile_url=mo.ui.text(
                label="Embedding server URL",
                value="http://localhost:8080/embedding",
                full_width=True
            ),
            offline_mode=mo.ui.checkbox(label="Run in offline mode"),
        )
        .form(show_clear_button=True, bordered=True)
    )


    def invalid_form(form):
        """A form (e.g. login) is invalid if it has no value,
        or if any of its keys have no value."""
        if form.value is None:
            return True

        for k in form.value.keys():
            if form.value[k] is None:
                return True

        return False
    return auth_form, invalid_form


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
