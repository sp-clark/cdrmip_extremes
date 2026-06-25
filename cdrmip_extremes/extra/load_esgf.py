import os
import sys
import xarray as xr
import pandas as pd
import cftime
import zarr
import intake
import fsspec
import requests
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def esgf_search(
    # server="https://esgf.nci.org.au/esg-search/search",
    # server="https://esgf-data.dkrz.de/esg-search",
    # server= "https://esgf-node.ornl.gov/esg-search/search",
    server="https://esgf-node.llnl.gov/esg-search/search",
    files_type="OPENDAP",
    local_node=True,
    project="CMIP6",
    verbose=False,
    format="application%2Fsolr%2Bjson",
    use_csrf=False,
    **search,
):
    """
    Function to search for available files on the ESGF servers, returned as pandas DataFrame.
    Adapted from: http://gallery.pangeo.io/repos/pangeo-gallery/cmip6/search_and_load_with_esgf_opendap.html"
    """
    client = requests.session()
    payload = search
    payload["project"] = project
    payload["type"] = "File"
    if local_node:
        payload["distrib"] = "false"
    if use_csrf:
        client.get(server)
        if "csrftoken" in client.cookies:
            # Django 1.6 and up
            csrftoken = client.cookies["csrftoken"]
        else:
            # older versions
            csrftoken = client.cookies["csrf"]
        payload["csrfmiddlewaretoken"] = csrftoken

    payload["format"] = format

    offset = 0
    numFound = 10000
    all_files = []
    files_type = files_type.upper()
    while offset < numFound:
        payload["offset"] = offset
        url_keys = []
        for k in payload:
            url_keys += ["{}={}".format(k, payload[k])]

        url = "{}/?{}".format(server, "&".join(url_keys))
        print(url)
        r = client.get(url)
        r.raise_for_status()
        resp = r.json()["response"]
        numFound = int(resp["numFound"])
        resp = resp["docs"]
        offset += len(resp)
        # for d in resp:
        #     if verbose:
        #         for k in d:
        #             print("{}: {}".format(k,d[k]))
        #     url = d["url"]
        #     for f in d["url"]:
        #         sp = f.split("|")
        #         if sp[-1] == files_type:
        #             all_files.append(sp[0].split(".html")[0])
        for d in resp:
            for f in d["url"]:
                sp = f.split("|")
                if sp[-1] == files_type:
                    url_path = sp[0].split(".html")[0]
                    # Assume version folder is just before the filename
                    version = url_path.split("/")[-2]
                    all_files.append(
                        {
                            "activity_id": d.get("activity_id", None)[0],
                            "experiment_id": d.get("experiment_id", None)[0],
                            "source_id": d.get("source_id", None)[0],
                            "member_id": d.get("member_id", None)[0],
                            "table_id": d.get("table_id", None)[0],
                            "variable_id": d.get("variable_id", None)[0],
                            "grid_label": d.get("grid_label", None)[0],
                            "version": version,
                            "url": url_path,
                        }
                    )
    # Use a dict to group URLs by dataset key
    grouped = defaultdict(list)
    for entry in all_files:
        key = tuple(
            entry[k]
            for k in [
                "experiment_id",
                "source_id",
                "member_id",
                "table_id",
                "variable_id",
            ]
        )
        grouped[key].append(entry["url"])  # collect URLs

    # Now create a DataFrame with one row per unique dataset key
    df_rows = []
    for key, urls in grouped.items():
        row = dict(
            zip(
                ["experiment_id", "source_id", "member_id", "table_id", "variable_id"],
                key,
            )
        )
        # take other metadata from the first entry
        first_entry = next(
            e
            for e in all_files
            if all(
                e[k] == val
                for k, val in zip(
                    [
                        "experiment_id",
                        "source_id",
                        "member_id",
                        "table_id",
                        "variable_id",
                    ],
                    key,
                )
            )
        )
        row["activity_id"] = first_entry["activity_id"]
        row["grid_label"] = first_entry["grid_label"]
        row["version"] = first_entry["version"]
        row["urls"] = urls  # list of URLs
        df_rows.append(row)

    df = pd.DataFrame(df_rows)

    # # convert to pandas dataframe with files sorted by version
    # # and version duplicates dropped
    # unique_keys = ["experiment_id", "source_id", "member_id", "table_id", "variable_id"]
    # # df = pd.DataFrame(all_files).sort_values(by='version',ascending=False).drop_duplicates(subset=unique_keys,keep='first')
    # df = pd.DataFrame(all_files)
    return df


def load_from_search(search_df,chunks,vars_to_drop):
    data = {}
    for index, row in tqdm(search_df.iterrows()):
        member_id = row["member_id"]
        path = row["urls"]
        data[member_id] = xr.open_mfdataset(
            path,
            chunks=chunks,
            drop_variables=vars_to_drop,
            use_cftime=True,
            combine="by_coords",
        )
    return data

def load_mmm_from_search(search_df,chunks,vars_to_drop):
    data = {}
    for index, row in tqdm(search_df.iterrows()):
        source_id = row["source_id"]
        print(source_id)
        version = row['version']
        urls = row["urls"]
        urls_to_load = []
        for url in urls:
            if version in url:
                urls_to_load.append(url)
        data[source_id] = xr.open_mfdataset(
            urls_to_load,
            chunks=chunks,
            drop_variables=vars_to_drop,
            use_cftime=True,
            combine="by_coords"
        )
    return data


# def load_mmm_from_search(search_df,chunks):
#     data = {}
#     for index, row in tqdm(search_df.iterrows()):
#         source_id = row["source_id"]
#         path = row["urls"]
#         data[source_id] = xr.open_mfdataset(
#             path,
#             chunks=chunks,
#             use_cftime=True,
#             combine="by_coords"
#         )
#     return data


def search_for_parent(ds_dict):
    search_dfs = []
    for member_id, ds in ds_dict.items():
        parent_id = ds.attrs["parent_variant_label"]
        df = esgf_search(
            activity_id=ds.attrs["parent_activity_id"],
            experiment_id=ds.attrs["parent_experiment_id"],
            source_id=ds.attrs["parent_source_id"],
            member_id=ds.attrs["parent_variant_label"],
            table_id=ds.attrs["table_id"],
            variable_id=ds.attrs["variable_id"],
        )
        search_dfs.append(df)
    return pd.concat(search_dfs)

def search_mmm_for_parent(ds_dict):
    search_dfs = []
    for source_id, ds in ds_dict.items():
        parent_id = ds.attrs["parent_variant_label"]
        df = esgf_search(
            activity_id=ds.attrs["parent_activity_id"],
            experiment_id=ds.attrs["parent_experiment_id"],
            source_id=ds.attrs["parent_source_id"],
            member_id=ds.attrs["parent_variant_label"],
            table_id=ds.attrs["table_id"],
            variable_id=ds.attrs["variable_id"],
        )
        search_dfs.append(df)
    return pd.concat(search_dfs)

def search_and_load_parent(ds_dict,chunks,vars_to_drop):
    search_df = search_mmm_for_parent(ds_dict)
    return load_mmm_from_search(search_df,chunks,vars_to_drop)


def concat_branches(data,ds_child, branch_date=None):
    """
    Function to concatenate experiment branches

    Arguments
    --------
    data: Dict
        Nested dictionary with first layer specifying model experiment and 
        second specifying member_id
    ds_child: xr.Dataset
        Dataset that we want to concatenate the parent branch to
    branch_date: str
        String specifying the cftime formatted branch-date if
        we know the ds_child.attrs['branch_time_in_parent'] to be incorrect.
        If None, will find branch_date from ds_child.attrs
    Returns
    ________
    xr.Dataset
        Dataset containing the concatenated parent and child datasets
    """
    parent_variant_label = ds_child.attrs["parent_variant_label"]
    parent_experiment_id = ds_child.attrs["parent_experiment_id"]
    ds_parent = data[parent_experiment_id][parent_variant_label]

    # get parent time and calendar
    parent_times = ds_parent.time
    calendar = parent_times.attrs.get("calendar", "standard")  # default to gregorian

    # determine branch date
    if branch_date is None:
        # convert branch_time_in_parent (days) to datetime
        branch_days = ds_child.branch_time_in_parent
        units = ds_child.parent_time_units
        branch_date = cftime.num2date(branch_days, units=units, calendar=calendar)

    # slice parent from branch_index onward
    ds_parent_slice = ds_parent.sel(time=slice(None, branch_date))

    # concatenate along time
    ds_concat = xr.concat([ds_parent_slice, ds_child], dim="time")

    return ds_concat

def concat_branches_mmm(data,ds_child, branch_date=None):
    """
    Function to concatenate experiment branches

    Arguments
    --------
    data: Dict
        Nested dictionary with first layer specifying model experiment and 
        second specifying member_id
    ds_child: xr.Dataset
        Dataset that we want to concatenate the parent branch to
    branch_date: str
        String specifying the cftime formatted branch-date if
        we know the ds_child.attrs['branch_time_in_parent'] to be incorrect.
        If None, will find branch_date from ds_child.attrs
    Returns
    ________
    xr.Dataset
        Dataset containing the concatenated parent and child datasets
    """
    parent_source_id = ds_child.attrs["parent_source_id"]
    parent_experiment_id = ds_child.attrs["parent_experiment_id"]
    ds_parent = data[parent_experiment_id][parent_source_id]

    # get parent time and calendar
    parent_times = ds_parent.time
    calendar = parent_times.attrs.get("calendar", "standard")  # default to gregorian

    # determine branch date
    if branch_date is None:
        # convert branch_time_in_parent (days) to datetime
        branch_days = ds_child.branch_time_in_parent
        units = ds_child.parent_time_units
        branch_date = cftime.num2date(branch_days, units=units, calendar=calendar)

    # slice parent from branch_index onward
    ds_parent_slice = ds_parent.sel(time=slice(None, branch_date))

    # concatenate along time
    ds_concat = xr.concat([ds_parent_slice, ds_child], dim="time")

    return ds_concat