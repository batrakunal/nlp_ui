import wikipedia
import wptools
from components.Text_cleaning import lower, spacing, eliminate_punctuation, eliminate_stopwords,text_cleaner
import pandas as pd

def founded_time(company_name=""):
    """
    This function will return the date of foundation of a given company.
    
    Parameters:
    company_name: pass a string containing the company name (make sure to include).
    """
    so = wptools.page(company_name).get_parse()
    df = pd.DataFrame(so.data["infobox"].values(), index=so.data["infobox"].keys())
    raw_d = df[0]["founded"].split("|")[1:]
    date = []
    for item in raw_d:
        temp_i = []
        for charac in list(item):
            if charac.isdigit() is True:
                temp_i.append(charac)
        new_i = "".join(temp_i)
        date.append(new_i)
    date = [i for i in date if i != ""]    
    return date


def num_employees(company_name=""):
    """
    This function will return the number of employees of a given company.
    
    Parameters:
    company_name: pass a string containing the company name (make sure to include).
    """
    so = wptools.page(company_name).get_parse()
    df = pd.DataFrame(so.data["infobox"].values(), index=so.data["infobox"].keys())
    raw = df[0]["num_employees"]
    cleaned = text_cleaner(raw, remove_stopwords=False).replace(" ","")
    return int(cleaned)


def other_raw_metadata(company_name=""):
    """
    This function will return the wikipedia metadata of a given company.

    Parameters:
    company_name: pass a string containing the company name (make sure to include).
    """
    so = wptools.page(company_name).get_parse()
    df = pd.DataFrame(so.data["infobox"].values(), index=so.data["infobox"].keys())
    return df
