import marimo as mo


def invalid_form(form):
    """A form (e.g. login) is invalid if it has no value,
    or if any of its keys have no value."""
    if form.value is None:
        return True

    for k in form.value.keys():
        if form.value[k] is None:
            return True

    return False


# Login form
login_form = mo.md("""
            # Login with Mastodon credentials
                
            {login}         {auth_code}
            """).batch(
                login=mo.ui.text(label="Login:"),
                auth_code=mo.ui.text(label="Authorization code:", kind="password"),
            ).form(show_clear_button=True, bordered=True)


# Configuration form
config_form = (mo.md("""
    # Configuration

    **Timelines**

    {tl_home} {tl_local} {tl_public}

    {tl_hashtag} {tl_hashtag_txt} {tl_list} {tl_list_txt}

    **Embeddings**

    {emb_server}

    {emb_server_url}

    {emb_server_model}

    **Caching**

    {offline_mode}
    """).batch(
        tl_home=mo.ui.checkbox(label="Home", value=True),
        tl_local=mo.ui.checkbox(label="Local"),
        tl_public=mo.ui.checkbox(label="Public"),
        tl_hashtag=mo.ui.checkbox(label="Hashtag"),
        tl_list=mo.ui.checkbox(label="List"),
        tl_hashtag_txt=mo.ui.text(),
        tl_list_txt=mo.ui.text(),
        emb_server=mo.ui.radio(label="Server type:",
                               options=["llamafile", "ollama"],
                               value="llamafile",
                               inline=True
                              ),
        emb_server_url=mo.ui.text(
            label="Embedding server URL:",
            value="http://localhost:8080/embedding",
            full_width=True
        ),
        emb_server_model=mo.ui.text(
            label="Embedding server model:",
            value="all-minilm"
        ),
        offline_mode=mo.ui.checkbox(label="Run in offline mode"),
    ).form(show_clear_button=True, bordered=True)
)


# Registration form
default_api_base_url = "https://your.instance.url"

reg_form = (mo.md("""
    # App Registration
    **Register your application**

    {application_name}
    {api_base_url}

    """).batch(
        application_name=mo.ui.text(
            label="Application name:",
            value="my_timeline_algorithm",
            full_width=True
        ),
        api_base_url=mo.ui.text(
            label="Mastodon instance API base URL:",
            value=default_api_base_url,
            full_width=True
        ),
    ).form(show_clear_button=True, bordered=True)
)

def invalid_reg_form(reg_form):
    """A reg form is invalid if the URL is the default one"""
    return invalid_form(reg_form) or reg_form.value['api_base_url']==default_api_base_url


def show_if(condition: bool, if_true, if_false):
    if condition:
        return if_true
    else:
        return if_false