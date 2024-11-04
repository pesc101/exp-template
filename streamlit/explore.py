"""Streamlit app to explore RAG evaluation results."""

import glob
import os
from json import JSONDecodeError
from typing import Dict, List

import pandas as pd
from encourage.utils.file_manager import FileManager

import streamlit as st

st.set_page_config(page_title="rag-evaluation", layout="wide")
st.title("🦙💭 RAG Evaluation Overview")

# Constants
ROOT_FOLDER = "./outputs"


def get_json_log_files(root_folder: str) -> List[str]:
    return sorted(glob.glob(f"{root_folder}/*/*.json*"), reverse=True)


def get_short_names(files: List[str]) -> List[str]:
    short_names = [
        os.path.join(os.path.basename(os.path.dirname(file_name)), os.path.basename(file_name))
        for file_name in files
    ]
    return [path.replace("/inference_log.json", "") for path in short_names]


def get_hydra_config(root_folder: str) -> List[str]:
    return sorted(glob.glob(f"{root_folder}/*/.hydra/config.yaml"), reverse=True)


@st.cache_data
def get_data(files_dict: Dict[str, str], short_name: str) -> pd.DataFrame:
    file_path = files_dict.get(short_name) or ""
    try:
        data = FileManager(file_path).load_jsonlines()
    except JSONDecodeError:
        try:
            data = FileManager(file_path).load_json()
        except JSONDecodeError as e:
            data = []
            st.error(f"Error: {e}")
    return pd.DataFrame(data[0])


def get_hydra_config_index(
    files_dict: Dict[str, str], short_name: str, hydra_config: List[str]
) -> int:
    folder_name = os.path.basename(os.path.dirname(files_dict[short_name]))
    for idx, config_path in enumerate(hydra_config):
        if folder_name in config_path:
            return idx
    return 0


def display_metadata(data: pd.DataFrame, question_index: int) -> None:
    with st.expander("Meta Data"):
        st.write(f"Request ID: {data.iloc[question_index]['request_id']}")
        st.write(f"Prompt ID: {data.iloc[question_index]['prompt_id']}")
        st.write(f"Conversation ID: {data.iloc[question_index]['conversation_id']}")
        st.write(f"Processing Time: {round(data.iloc[question_index]['processing_time'], 2)}ms")


def display_prompts(data: pd.DataFrame, question_index: int) -> None:
    with st.expander("Prompts", expanded=True):
        st.markdown("**<span style='font-size:20px'>Sys Prompt:</span>**", unsafe_allow_html=True)
        st.code(data.iloc[question_index]["sys_prompt"], line_numbers=True, language="markdown")
        st.markdown("**<span style='font-size:20px'>User Prompt:</span>**", unsafe_allow_html=True)
        st.write(data.iloc[question_index]["user_prompt"])


def display_context(data: pd.DataFrame, question_index: int) -> None:
    with st.expander("Context", expanded=False):
        st.markdown("**<span style='font-size:20px'>Context:</span>**", unsafe_allow_html=True)
        st.write(data.iloc[question_index]["context"]["context"])


def display_response(data: pd.DataFrame, question_index: int) -> None:
    with st.expander("Response", expanded=True):
        st.markdown("**<span style='font-size:20px'>Ground True:</span>**", unsafe_allow_html=True)
        st.write(data.iloc[question_index]["meta_data"]["answers"])
        st.markdown("**<span style='font-size:20px'>Response:</span>**", unsafe_allow_html=True)
        st.write(data.iloc[question_index]["response"])


# Main logic
files = get_json_log_files(ROOT_FOLDER)
short_names = get_short_names(files)
files_dict = dict(zip(short_names, files))
hydra_config = get_hydra_config(ROOT_FOLDER)

with st.sidebar:
    short_name = st.selectbox("Select File", options=short_names, index=0) or ""
    data = get_data(files_dict, short_name)
    hydra_config_index = get_hydra_config_index(files_dict, short_name, hydra_config)
    config = FileManager(hydra_config[hydra_config_index]).load_yaml()

    question_select = ""
    if not data.empty:
        question_content_list = data["user_prompt"].dropna().unique().tolist()
        question_select = st.selectbox("Select Question", question_content_list, index=0)
        question_index = data[data["user_prompt"] == question_select].index[0]

with st.expander("Run Config"):
    st.write(config)

with st.expander("Raw Data"):
    st.write(data.iloc[question_index])

display_metadata(data, question_index)
display_prompts(data, question_index)
display_context(data, question_index)
display_response(data, question_index)
