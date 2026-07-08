"""
To build curl commands from copy pasted forms from the biobank website
"""

import sys

FORM_TEXT = """
            <form name="fetch" action="https://biota.osc.ox.ac.uk/dataset.cgi" method="post">

<input type="hidden" name="id" value="67343563">
<input type="hidden" name="s" value="38023627">
<input type="hidden" name="t" value="1262365114">
<input type="hidden" name="i" value="76.66.198.53">
<input type="hidden" name="v" value="dc7d7089413fa4a56c6s301a059148asa81904816804130e7909ec72402">
<input class="btn_glow" type="submit" value="Fetch">
</form>
"""


NAME = "DOWNLOAD.enc"  # Downloaded file's name


test = """
<form name="fetch" action="https://biota.osc.ox.ac.uk/dataset.cgi" method="post">
<input type="hidden" name="id" value="671600"/>
<input type="hidden" name="s" value="305736"/>
<input type="hidden" name="t" value="1684501586"/>
<input type="hidden" name="i" value="67.244.49.54"/>
<input type="hidden" name="v" value="891f3ec7f3388d4c7a0c094ef1abde73f44c356f2732dade6a7921d9770dd095"/>
<input class="btn_glow" type="submit" value="Fetch"/>
</form>
"""


def get_fields(txt):
    i = txt.find('''name="fetch"''')
    if i == -1:
        print('Fetch form not in text')
        return
    action, i = get_field(txt, i, '''action="''')
    fields = {'action': action}
    for field in ['id', 's', 't', 'i', 'v']:
        fields[field], i = get_field(txt, i)
    return fields


def get_field(txt, start, target='''value="'''):
    start = txt.find(target, start)
    end = txt.find('''"''', start + len(target))
    return txt[start + len(target): end], end


def fields_to_curl(name, action, id, s, t, i, v):
    return f"""
curl -d "id={id}&s={s}&t={t}&i={i}&v={v}&submit=Fetch" \
-X POST {action} \
-o {name}
    """


def txt_to_curl(name, txt):
    return fields_to_curl(name, **get_fields(txt))

# check to see if an argument was provided (single argument with path to form text in a file)
if len(sys.argv) > 1:
    try:
        with open (sys.argv[1], "r") as form_text_file:
            FORM_TEXT = form_text_file.read()
    except:
        print(f'This program expects the input argument, if provided, to be a path')
        print(f'to a file containing the form data from the ukbiobank website.')
        exit(1)

print(txt_to_curl(NAME, FORM_TEXT))
