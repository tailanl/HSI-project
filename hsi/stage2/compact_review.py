"""Compact boolean transport; no check, evidence or threshold is removed."""


def schema(checks,reason_limit=100):
    return {"type":"object","additionalProperties":False,"properties":{
        "checks":{"type":"array","items":{"type":"boolean"},"minItems":len(checks),"maxItems":len(checks)},
        "confidence":{"type":"number","minimum":0,"maximum":1},
        "reason":{"type":"string","maxLength":reason_limit}},"required":["checks","confidence","reason"]}


def normalize(value,checks,reason_limit=100):
    if not isinstance(value,dict) or set(value)!={"checks","confidence","reason"}:
        raise ValueError("Unexpected compact audit fields")
    flags=value["checks"]
    if not isinstance(flags,list) or len(flags)!=len(checks) or any(type(v) is not bool for v in flags):
        raise ValueError("Compact audit must contain one boolean per ordered check")
    if len(checks)!=len(set(checks)):
        raise ValueError("Duplicate audit check name")
    if not isinstance(value["reason"],str) or len(value["reason"])>reason_limit:
        raise ValueError("Compact audit reason too long")
    return {**dict(zip(checks,flags)),"confidence":value["confidence"],"reason":value["reason"]}


def instruction(checks):
    return (" Return checks as an ordered boolean array with exactly "+str(len(checks))+" elements, in this exact order: "
        +", ".join(checks)+". Return confidence separately, and one brief reason of at most 12 words and 100 characters. "
        "The array only shortens the output notation; inspect every check independently. No coordinates. Return only JSON.")
