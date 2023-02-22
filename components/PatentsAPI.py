"""
US PTO API

@author: Pedro Veronezi
@date: 22 October 2018

"""

from uspto.peds.client import UsptoPatentExaminationDataSystemClient
from uspto.pbd.client import UsptoPairBulkDataClient
from datetime import datetime
import pypatent
import pandas as pd
import datetime


def bulk_data(
    applicant_name=None,
    patent_title=None,
    application_status=None,
    date_filter_start=None,
    date_filter_finish=None,
):
    """
    Function responsible to search on the US PTO Database, given the arguments. The first 3 parameters needs to have at
    least one argument. The function returns a dictionary with the required information.
    More info at: https://docs.ip-tools.org/uspto-opendata-python/pbd.html
    :param applicant_name: basestring Declares the patent's applicant name
    :param patent_title: basestring Declares the patent's title name
    :param application_status: basestring Define the application status: possible values ['patented', 'pending]
    :param date_filter_start: basestring Represents the startdate for the search YYYY-MM-DD fomart
    :param date_filter_finish: base string Represents the enddate for the search. YYYY-MM-DD format
    :return: a python dictionary with all the retrieved information
    """
    client = UsptoPairBulkDataClient()
    expression = ""
    filter_patent = ""

    # If scenarios to define the expression
    if applicant_name is None and patent_title is None and application_status is None:
        print(
            "You have to input at least 1 from the 3 first arguments, function will return None"
        )
        return None

    if applicant_name is not None:
        expression += "firstNamedApplicant:(%s)" % (applicant_name,)

    if patent_title is not None:
        if len(expression) > 1:
            expression += "AND patentTitle:(%s)" % (patent_title,)
        else:
            expression += "patentTitle:(%s)" % (patent_title,)

    if application_status is not None:
        if len(expression) > 1:
            expression += "AND appStatus:(%s)" % (application_status,)
        else:
            expression += "appStatus:(%s)" % (application_status,)

    if date_filter_start is None and date_filter_finish is None:
        print("No filter defined")

    if date_filter_start is not None and date_filter_finish is None:
        filter_patent += (
            "appFillingDate:[%sT00:00:00Z TO "
            + datetime.now().strftime("%Y-%M-%dT%H:%M:%SZ")
            + "]" % (date_filter_start,)
        )

    if date_filter_start is not None and date_filter_finish is not None:
        filter_patent += "appFillingDate:[%sT00:00:00Z TO %sT00:00:00Z]" % (
            date_filter_start,
            date_filter_finish,
        )

    return client.search(expression, filter=filter_patent, start=0, rows=20)


def download_patent(document_id_number=None):
    """
    The download interface uses the search interface and adds automation for requesting and downloading package bundles
    for search results as outlined in the »API Tutorial« section of the API documentation.
    :param document_id_number: basestring that identifies the patent number and downlaod teh document
    :param bul_request_result: a dict that results from the function 'bulk_data'
    :return:
    """
    if document_id_number is None:
        print("You must pass a valid argument")
        return None
    else:
        client = UsptoPairBulkDataClient()
        return client.download_document(document_id_number)


def patent_examination_data(
    applicant_name=None,
    patent_title=None,
    application_status=None,
    date_filter_start=None,
    date_filter_finish=None,
):
    """
    The PEDS system provides additional information concerning the transaction activity that has occurred for each
    patent. The transaction history includes the transaction date, transaction code and transaction description for
    each transaction activity.
    More info at: https://docs.ip-tools.org/uspto-opendata-python/peds.html
    :param applicant_name: basestring Declares the patent's applicant name
    :param patent_title: basestring Declares the patent's title name
    :param application_status: basestring Define the application status: possible values ['patented', 'pending]
    :param date_filter_start: basestring Represents the startdate for the search YYYY-MM-DD fomart
    :param date_filter_finish: base string Represents the enddate for the search. YYYY-MM-DD format
    :return: a python dictionary with all the retrieved information
    """
    client = UsptoPatentExaminationDataSystemClient()
    expression = ""
    filter_patent = ""

    # If scenarios to define the expression
    if applicant_name is None and patent_title is None and application_status is None:
        print(
            "You have to input at least 1 from the 3 first arguments, function will return None"
        )
        return None

    if applicant_name is not None:
        expression += "firstNamedApplicant:(%s)" % (applicant_name,)

    if patent_title is not None:
        if len(expression) > 1:
            expression += "AND patentTitle:(%s)" % (patent_title,)
        else:
            expression += "patentTitle:(%s)" % (patent_title,)

    if application_status is not None:
        if len(expression) > 1:
            expression += "AND appStatus:(%s)" % (application_status,)
        else:
            expression += "appStatus:(%s)" % (application_status,)

    if date_filter_start is None and date_filter_finish is None:
        print("No filter defined")

    if date_filter_start is not None and date_filter_finish is None:
        filter_patent += (
            "appFillingDate:[%sT00:00:00Z TO "
            + datetime.now().strftime("%Y-%M-%dT%H:%M:%SZ")
            + "]" % (date_filter_start,)
        )

    if date_filter_start is not None and date_filter_finish is not None:
        filter_patent += "appFillingDate:[%sT00:00:00Z TO %sT00:00:00Z]" % (
            date_filter_start,
            date_filter_finish,
        )

    return client.search(expression, filter=filter_patent, start=0, rows=20)


def patent_extractor(keyword, start_date, end_date):
    """
    The function to extract patents from USPTO based on the keywords specified by the user.

    :param keyword: The topic of interest to extract the patents
    :param start_date: Start date for the
    :param end_date:
    :return:
    """
    a = pypatent.Search(keyword).as_dataframe()

    patent_info = a[["title", "patent_date", "abstract", "description", "url"]]

    patent_info["patent_date"] = patent_info["patent_date"].apply(
        lambda x: datetime.datetime.strptime(x, "%B %d,  %Y")
    )

    c = patent_info[
        (
            patent_info["patent_date"]
            >= datetime.datetime.strptime(start_date, "%Y-%m-%d")
        )
        & (
            patent_info["patent_date"]
            <= datetime.datetime.strptime(end_date, "%Y-%m-%d")
        )
    ]

    return c
