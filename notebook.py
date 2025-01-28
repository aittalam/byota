import marimo

__generated_with = "0.10.17"
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
    from pathlib import Path

    from byota.embeddings import (
        EmbeddingService,
        LLamafileEmbeddingService,
        OllamaEmbeddingService
    )
    import byota.mastodon as byota_mastodon
    from byota.search import SearchService
    return (
        BeautifulSoup,
        EmbeddingService,
        LLamafileEmbeddingService,
        OllamaEmbeddingService,
        Path,
        SearchService,
        TSNE,
        alt,
        byota_mastodon,
        functools,
        mo,
        pd,
        pickle,
        requests,
        time,
    )


@app.cell
def _(Path):
    # internal variables

    # client and user credentials filenames
    clientcred_filename = "secret_clientcred.txt"
    usercred_filename = "secret_usercred.txt"

    # dump files for offline mode
    paginated_data_file = "dump_paginated_data.pkl"
    dataframes_data_file = "dump_dataframes.pkl"
    embeddings_data_file = "dump_embeddings.pkl"

    cached_timelines = True
    cached_dataframes = True
    cached_embeddings = True

    app_registered = True if Path(clientcred_filename).is_file() else False
    return (
        app_registered,
        cached_dataframes,
        cached_embeddings,
        cached_timelines,
        clientcred_filename,
        dataframes_data_file,
        embeddings_data_file,
        paginated_data_file,
        usercred_filename,
    )


@app.cell
def _(app_registered, mo, reg_form, show_if):
    show_if(not app_registered, reg_form, mo.md("**Your application is registered**"))
    return


@app.cell
def _(
    app_registered,
    byota_mastodon,
    clientcred_filename,
    invalid_form,
    mo,
    reg_form,
):
    if not app_registered:
        mo.stop(invalid_form(reg_form), mo.md("**Invalid values provided in the registration form**"))

        byota_mastodon.register_app(
            reg_form.value['application_name'],
            reg_form.value['api_base_url'],
            clientcred_filename
        )
    return


@app.cell
def _(auth_form):
    auth_form
    return


@app.cell
def _(
    LLamafileEmbeddingService,
    auth_form,
    byota_mastodon,
    clientcred_filename,
    invalid_form,
    mo,
    timelines_dict,
    usercred_filename,
):
    # check for anything invalid in the form
    mo.stop(invalid_form(auth_form),
            mo.md("**Submit the form to continue.**"))

    # login (and break if that does not work)
    mastodon_client = byota_mastodon.login(clientcred_filename,
                                           usercred_filename,
                                           auth_form.value.get("login"),
                                           auth_form.value.get("pw")
                                          )
    mo.stop(mastodon_client is None,
            mo.md("**Authentication error.**"))

    # instatiate an embedding service (and break if it does not work)
    embedding_service = LLamafileEmbeddingService(
        auth_form.value["emb_llamafile_url"]
    )

    mo.stop(
        not embedding_service.is_working(),
        mo.md("**Cannot access llamafile embedding server.**"),
    )

    # collect the names of the timelines we want to download from
    timelines = []
    for k in timelines_dict.keys():
        if auth_form.value[k]:
            tl_string = timelines_dict[k]
            if tl_string in ["tag", "list"]:
                tl_string += f'/{auth_form.value[f"{k}_txt"]}'
            timelines.append(tl_string)

    # set offline mode
    offline_mode = auth_form.value["offline_mode"]
    return (
        embedding_service,
        k,
        mastodon_client,
        offline_mode,
        timelines,
        tl_string,
    )


@app.cell
def _(mo):
    mo.md(r"""# Getting data from my Mastodon account...""")
    return


@app.cell
def _(
    build_cache_dataframes,
    build_cache_paginated_data,
    cached_dataframes,
    cached_timelines,
    dataframes_data_file,
    mastodon_client,
    mo,
    paginated_data_file,
    timelines,
):
    paginated_data = build_cache_paginated_data(mastodon_client,
                                                timelines,
                                                cached_timelines,
                                                paginated_data_file)
    mo.stop(paginated_data is None, mo.md(f"**Issues connecting to Mastodon**"))


    dataframes = build_cache_dataframes(paginated_data,
                                         cached_dataframes,
                                         dataframes_data_file)

    mo.stop(paginated_data is None, mo.md(f"**Issues connecting to Mastodon**"))
    return dataframes, paginated_data


@app.cell
def _(mo):
    mo.md("""# My timeline""")
    return


@app.cell
def _(
    build_cache_embeddings,
    cached_embeddings,
    dataframes,
    embedding_service,
    embeddings_data_file,
):
    # calculate embeddings
    embeddings = build_cache_embeddings(embedding_service,
                                        dataframes,
                                        cached_embeddings,
                                        embeddings_data_file)
    return (embeddings,)


@app.cell
def _(TSNE, alt, dataframes, embeddings, mo, pd):
    import numpy as np

    def tsne(dataframes, embeddings, perplexity, random_state=42):
        """Runs dimensionality reduction using TSNE on the input embeddings.
        Returns dataframes containing posts id, text, and 2D coordinates
        for plotting.
        """
        tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)

        all_embeddings = np.concatenate([v for v in embeddings.values()])
        all_projections = tsne.fit_transform(all_embeddings)

        dfs = []
        start_idx = 0
        end_idx = 0
        for kk in embeddings:
            end_idx+=len(embeddings[kk])
            df = dataframes[kk]
            df["x"] = all_projections[start_idx:end_idx, 0]
            df["y"] = all_projections[start_idx:end_idx, 1]
            df["label"] = kk
            dfs.append(df)
            start_idx=end_idx

        return pd.concat(dfs, ignore_index=True), all_embeddings


    df_, all_embeddings = tsne(dataframes, embeddings, perplexity=16)

    chart = mo.ui.altair_chart(
        alt.Chart(df_, title="Timeline Visualization")#, height=800)
        .mark_point()
        .encode(
            x="x",
            y="y",
            color="label"
        )
    )
    return all_embeddings, chart, df_, np, tsne


@app.cell
def _(chart, mo):
    mo.vstack(
        [
            chart,
            chart.value[["id", "label", "text"]] if len(chart.value) > 0 else chart.value,
        ]
    )
    return


@app.cell
def _(query_form):
    query_form
    return


@app.cell
def _(SearchService, all_embeddings, df_, embedding_service, query_form):
    search_service = SearchService(all_embeddings, embedding_service)
    indices = search_service.most_similar_indices(query_form.value)
    df_.iloc[indices][['label','text']]
    return indices, search_service


@app.cell
def _(all_embeddings, np, query_form, search_service):
    import matplotlib.pyplot as plt

    mse = search_service.most_similar_embeddings(query_form.value)
    diff_small = mse[0]-mse[1]
    diff_mid = mse[0]-mse[4]
    diff_large = mse[0]-all_embeddings[42]

    plt.rcParams["figure.figsize"] = (20,3)
    plt.plot(diff_large)
    plt.plot(diff_small)
    plt.legend([f"Diff with a random embedding (norm={np.linalg.norm(diff_large):.2f})", 
                f"Diff with a similar embedding (norm={np.linalg.norm(diff_small):.2f})"])

    plt.show()
    return diff_large, diff_mid, diff_small, mse, plt


@app.cell
def _(mo):
    # Create the Configuration form

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

    # a dictionary mapping Timeline UI checkboxes with the respective
    # strings that identify them in the Mastodon API
    timelines_dict = {
        "tl_home": "home",
        "tl_local": "local",
        "tl_public": "public",
        "tl_hashtag": "tag",
        "tl_list": "list",
    }

    def invalid_form(form):
        """A form (e.g. login) is invalid if it has no value,
        or if any of its keys have no value."""
        if form.value is None:
            return True

        for k in form.value.keys():
            if form.value[k] is None:
                return True

        return False
    return auth_form, invalid_form, timelines_dict


@app.cell
def _(mo):
    # Create a registration form

    default_api_base_url = "https://your.instance.url"

    reg_form = (
        mo.md(
            """
        # App Registration
        **Register your application**

        {application_name}
        {api_base_url}

    """
        )
        .batch(
            application_name=mo.ui.text(
                label="Application name:",
                value="my_timeline_algorithm",
                full_width=True
            ),
            api_base_url=mo.ui.text(
                label="Mastodon instance API base URL",
                value=default_api_base_url,
                full_width=True
            ),
        )
        .form(show_clear_button=True, bordered=True)
    )

    def invalid_reg_form(reg_form):
        """A reg form is invalid if the URL is the default one"""
        if reg_form.value is None:
            return True

        for k in reg_form.value.keys():
            if reg_form.value[k] is None or reg_form.value[k]=="":
                return True

        if reg_form.value['api_base_url']==default_api_base_url:
            return True

        return False


    def show_if(condition: bool, if_true, if_false):
        if condition:
            return if_true
        else:
            return if_false
    return default_api_base_url, invalid_reg_form, reg_form, show_if


@app.cell
def _(mo):
    query_form = mo.ui.text(
        value="42",
        label="Enter a post id or some free-form text to find the most similar posts:\n",
        full_width=True,
    )
    return (query_form,)


@app.cell
def _(BeautifulSoup, EmbeddingService, byota_mastodon, pd, pickle, time):
    def build_cache_paginated_data(mastodon_client,
                                   timelines: list,
                                   cached: bool,
                                   paginated_data_file: str) -> dict[str,any]:
        """Given a list of timeline names and a mastodon client,
        use the mastodon client to get paginated data from each
        and return a dictionary that contains, for each key, all
        the retrieved data.
        If cached==True, the `paginated_data_file` file will be loaded.
        """
        if not cached:
            paginated_data = {}
            for tl in timelines:
                paginated_data[tl] = byota_mastodon.get_paginated_data(mastodon_client, tl)
            with open(paginated_data_file, "wb") as f:
                pickle.dump(paginated_data, f)

        else:
            print(f"Loading cached paginated data from {paginated_data_file}")
            with open(paginated_data_file, "rb") as f:
                paginated_data = pickle.load(f)

        return paginated_data


    def build_cache_dataframes(paginated_data: dict[str, any],
                               cached: bool,
                               dataframes_data_file: str) -> dict[str, any]:
        """Given a dictionary with paginated data from different timelines,
        return another dictionary that contains, for each timeline, a compact
        pandas DataFrame of (id, text) pairs.
        If cached==True, the `dataframes_data_file` file will be loaded.
        """
        if not cached:
            dataframes = {}
            for k in paginated_data:
                dataframes[k] = pd.DataFrame(
                    get_compact_data(paginated_data[k]), 
                    columns=["id", "text"]
                )
            with open(dataframes_data_file, "wb") as f:
                pickle.dump(dataframes, f)
        else:
            print(f"Loading cached dataframes from {dataframes_data_file}")
            with open(dataframes_data_file, "rb") as f:
                dataframes = pickle.load(f)

        return dataframes


    def build_cache_embeddings(embedding_service: EmbeddingService,
                               dataframes: dict[str, any],
                               cached: bool,
                               embeddings_data_file: str) -> dict[str, any]:
        """Given a dictionary with dataframes from different timelines,
        return another dictionary that contains, for each timeline, the
        respective embeddings calculated with the provided embedding service.
        If cached==True, the `embeddings_data_file` file will be loaded.
        """
        if not cached:
            embeddings = {}
            for k in dataframes:
                print(f"Embedding posts from timeline: {k}")
                tt_ = time.time()
                embeddings[k] = embedding_service.calculate_embeddings(dataframes[k]["text"])
                print(time.time() - tt_)
            with open(embeddings_data_file, "wb") as f:
                pickle.dump(embeddings, f)
        else:
            print(f"Loading cached embeddings from {embeddings_data_file}")
            with open(embeddings_data_file, "rb") as f:
                embeddings = pickle.load(f)

        return embeddings


    def get_compact_data(paginated_data: list) -> list[tuple[int, str]]:
        """Extract compact (id, text) pairs from a paginated list of posts."""
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
    return (
        build_cache_dataframes,
        build_cache_embeddings,
        build_cache_paginated_data,
        get_compact_data,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
