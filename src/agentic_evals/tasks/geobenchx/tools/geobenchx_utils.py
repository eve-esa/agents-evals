"""Dataframe description helper used by GeoBenchX's tools.

Copied from ``geobenchx/utils.py`` (Krechetova, MIT). ``get_solution_code`` and
``compute_confusion_stats`` stay in the upstream module; our metric code has its
own ``solution_code``, and the verbatim module used to check it is under
``tests/fixtures/geobenchx/upstream``.
"""

from io import StringIO


def get_dataframe_info(df):    
    """ Gets information from df.info() as a string"""    
    buffer = StringIO()
    df.info(buf=buffer)
    # # Get only string (object) columns
    # string_columns = df.select_dtypes(include=['object']).columns         
    # # Print unique values for each string column
    # for col in string_columns:
    #     print(f"\nUnique values in {col}:", file=buffer)
    #     print(str(df[col].unique()), file=buffer)

    # Calculate percentages of nonj-empty cells
    non_empty_stats = (df.count() / len(df) * 100).round(2)
    non_empty_info = "\n\nNon-empty values (%):\n"
    for col, pct in non_empty_stats.items():
        non_empty_info += f"{col}: {pct}%\n"

    info_string = buffer.getvalue()
    buffer.close() 
    return info_string, non_empty_info


