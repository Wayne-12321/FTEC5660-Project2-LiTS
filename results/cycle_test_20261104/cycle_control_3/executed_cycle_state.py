"""Conservative canonicalization of LLM state text; no transitions or planning."""
import re

def canonical_state(text, expected_blocks=None):
    if not isinstance(text,str):return None
    text=text.lower()
    assignments=[]
    assignments += [(b,'table') for b in re.findall(r'the (\w+) block is on the table',text)]
    assignments += re.findall(r'the (\w+) block is on top of the (\w+) block',text)
    assignments += [(b,'hand') for b in re.findall(r'the (\w+) block is in the hand',text)]
    pos={}
    for b,support in assignments:
        if b in pos and pos[b]!=support:return None
        pos[b]=support
    if not pos or (expected_blocks is not None and set(pos)!=set(expected_blocks)):return None
    held=[b for b,s in pos.items() if s=='hand']
    hand_claims=set(re.findall(r'hand is holding the (\w+) block',text))
    empty='hand is empty' in text
    if len(held)>1:return None
    if held:
        if empty or hand_claims!={held[0]}:return None
    elif not empty or hand_claims:return None
    supports=[s for s in pos.values() if s not in ('table','hand')]
    if any(s not in pos for s in supports) or len(supports)!=len(set(supports)):return None
    clear=set(re.findall(r'the (\w+) block is clear',text))
    expected_clear={b for b,s in pos.items() if b not in supports and s!='hand'}
    if clear!=expected_clear:return None
    for b in pos:
        seen=set();cur=b
        while cur not in ('table','hand'):
            if cur in seen:return None
            seen.add(cur);cur=pos[cur]
    return tuple(sorted(pos.items()))
