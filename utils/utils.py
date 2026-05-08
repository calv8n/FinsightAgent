import re


def tag_content(text, tags):
    """
    This Function extracts the content between HTML tags.
    Input Args:
    text: raw text containing html tags
    tags: list of tags for content extraction

    Output Args:
    res: Dictionary of tags and extracted content. If content found, it's returned with the content else with None for each tag
    """
    res = {}
    for tag in tags:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            res[tag] = match.group(1).strip()
        else:
            res[tag] = None
    return res
