"""
To build curl commands from copy pasted forms from the biobank website
"""


FORM_TEXT = """
            <form name="fetch" action="https://biota.osc.ox.ac.uk/dataset.cgi" method="post">

                <input type="hidden" name="id" value="671599"/>
                <input type="hidden" name="s" value="305736"/>
                <input type="hidden" name="t" value="1684504514"/>
                <input type="hidden" name="i" value="67.244.49.54"/>
                <input type="hidden" name="v" value="da5aa919c0119423d8335cf169f51bb2a834f2967558e3a45f0f49d0157d6428"/>
                <input class="btn_glow" type="submit" value="Fetch"/>
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


print(txt_to_curl(NAME, FORM_TEXT))
