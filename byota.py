from mastodon import Mastodon
from scipy import spatial
import marimo as mo
import pandas as pd
import requests
import json
import numpy as np


# -- Mastodon ---------------------------------------------------------------------

def get_mastodon_data(login: str, pw: str, timeline_type: str, max_pages: int = 40):
    """Authenticates to mastodon and gets paginated posts from
    one of the following timelines: ‘home’, ‘local’, ‘public’,
    ‘tag/hashtag’ or ‘list/id’.

    See https://mastodonpy.readthedocs.io/en/stable/07_timelines.html
    and https://docs.joinmastodon.org/methods/timelines/#home
    """
    mastodon = Mastodon(
        client_id="mastimeline_clientcred.secret",
    )

    try:
        mastodon.log_in(login, pw, to_file="mastimeline_usercred.secret")
        mastodon = Mastodon(
            access_token="mastimeline_usercred.secret",
        )
    except Exception as e:
        print(f"Mastodon auth error: {e}")
        return None

    tl = mastodon.timeline(timeline_type)
    # tl = mastodon.timeline_home()

    paginated_data = [tl]
    i = 0
    with mo.status.progress_bar(total=max_pages, 
                                title=f"Timeline: {timeline_type}") as bar:
        while len(tl) > 0 and i < max_pages:
            i += 1
            print(
                f"Loading page {i}: max_id = {tl._pagination_next.get('max_id')}"
            )
            tl = mastodon.timeline(timeline_type,
            # tl = mastodon.timeline_home(
                max_id=tl._pagination_next.get("max_id")
            )
            paginated_data.append(tl)
            bar.update()
    return paginated_data


# -- Embeddings -------------------------------------------------------------------

def get_llamafile_embedding(input_text: str, llamafile_URL: str):

    try:
        response = requests.request(
            url=llamafile_URL,
            method="POST",
            data={"content": input_text},
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        raise

    # print(f"Response text: {response.text}")
    return json.loads(response.text)["embedding"]


def get_ollama_embedding(input_text: str, ollama_URL: str, ollama_model: str):

    if not input_text:
        input_text = " "

    try:
        response = requests.request(
            url=ollama_URL,
            method="POST",
            data=json.dumps({
                "model": ollama_model,
                "prompt": input_text
            }),
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        raise

    # print(f"Response text: {response.text}")
    return json.loads(response.text)["embedding"]


# @mo.cache
def calculate_embeddings(texts, emb_func):
    embeddings = []
    for i, t in enumerate(texts):
        embeddings.append(emb_func(input_text=str(t)))
        if not (i % 10):
            print(".", end="")
    return np.array(embeddings)


# -- Similarity -------------------------------------------------------------------

def is_integer_string(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def most_similar_to(query, text, embeddings, llamafile_URL, k=5):

    if is_integer_string(query):
        query = int(query)
        print(f"Most {k} similar posts to:\n{text[query]}\n\n")
        query = embeddings[query]
    else:
        print(f"Most {k} similar posts to:\n{query}\n\n")
        query = get_llamafile_embedding(query, llamafile_URL)

    tree = spatial.KDTree(embeddings)

    # get the k nearest neighbors
    neighbors = []
    idx = tree.query(query, k=k + 1)[1][1:]
    for i in idx:
        neighbors.append((i, text[i]))
        print(f"{i}\n{text[i]}\n")

    return pd.DataFrame(neighbors, columns=["local_id", "text"])
