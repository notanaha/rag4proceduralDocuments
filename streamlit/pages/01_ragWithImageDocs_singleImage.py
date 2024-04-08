import os, json
from IPython.display import Image
import openai
from azure.storage.blob import BlobServiceClient
import streamlit as st
from utilities import utils

#--------------------------------------#
# Set OpenAI variables                 #
#--------------------------------------#
use_azure_active_directory = False  # Set this flag to True if you are using Azure Active Directory
if not use_azure_active_directory:
    aoai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    aoai_api_key = os.environ["AZURE_OPENAI_API_KEY"]
    api_version = os.environ["AZURE_OPENAI_API_VERSION"]

    client = openai.AzureOpenAI( 
        azure_endpoint=aoai_endpoint,
        api_key=aoai_api_key,
        api_version= api_version
    )
    embedding_model: str = "text-embedding-ada-002" 


#--------------------------------------#
# Set blob variables                   #
#--------------------------------------#
connection_string = os.environ["STORAGE_CONN_STR"]
# Temporarily set as identity access fails
storage_sas_token = os.environ["STORAGE_SAS_TOKEN"] 

container_name = "manual-test"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
split_word = "_"



def main():

    query = st.text_input("質問を入力してください:", value="普通預金の利息の決算日はいつですか")

    if st.button("Enter", key="confirm"):
        if query.lower() == "reset":
            for key in st.session_state.keys():
                del st.session_state[key]
            st.write("Conversation history has been reset.")
            return

        if 'messages' not in st.session_state:

            #--------------------------------------#
            # Retrieving answer from Search Index  #
            #--------------------------------------#
            st.write("Retrieving answer from Search Index.")
            results = utils.search_index(query, client, embedding_model)
            answer_context = []
            
            for result in results:
                titles_and_content = {}
                titles_and_content["title"] = result["title"]
                titles_and_content["content"] = result["content"]
                answer_context.append(titles_and_content)

            #--------------------------------------#
            # Retrieving answer from GPT-4         #
            #--------------------------------------#
            with open(os.path.join('utilities','system_message_02.txt'), "r", encoding = 'utf-8') as f:
                system_message = f.read()

            messages = [] 
            messages.append({"role": "system","content": system_message})

            content = {}
            content["question"] = query
            content["context"] = answer_context
            messages.append({"role": "user", "content": str(content)})

            st.write("Retrieving answer from GPT-4")
            print("Retrieving answer from GPT-4")

            response = utils.gpt4_turbo_query(messages, client)
            answer = response.choices[0].message.content

            answer = json.loads(answer)
            st.write(answer)
            print(answer)

            #--------------------------------------#
            # Get Image URL on Storage Account     #
            #--------------------------------------#
            blob_name = answer["title"]
            image_urls = utils.list_blobs_titles_and_urls(blob_service_client, container_name, blob_name, split_word)
            if len(image_urls) == 0:
                st.write("No images found.")
                return

            #--------------------------------------#
            # Retrieving answer from GPT-4V        #
            #--------------------------------------#
            with open(os.path.join('utilities','system_message_01.txt'), "r", encoding = 'utf-8') as f:
                system_message = f.read()

            messages = [] # reset messages for GPT-4V
            messages.append({"role": "system","content":[{"type": "text", "text": system_message}]})

            content = []
            content.append({"type": "text", "text": query})

            for url in image_urls:
                storagepath = url['title']
                storagepath_stem = os.path.splitext(storagepath)[0]
                if not (storagepath_stem[-3] == split_word and storagepath_stem[-2:].isdigit()):
                    storagepath_stem += split_word + '01' # storagepath doesn't have a number suffix
                if storagepath_stem == answer['page']:
                    content.append({"type": "text", "text": storagepath})
                    content.append({"type": "image_url", "image_url": url['url']+storage_sas_token})
                    break
                
            messages.append({"role": "user","content":content})

            st.write("Retrieving answer from GPT-4 with Vision")
            print("Retrieving answer from GPT-4 with Vision")

            response = utils.gpt4v_query(messages, aoai_endpoint, aoai_api_key, api_version)
            answer = response.json()["choices"][0]["enhancements"]["grounding"]["lines"][0]["text"]
            #answer = response.json()["choices"][0]["message"]["content"]
            
            #st.write(json.dumps(answer, indent=4, ensure_ascii=False))
            st.write(answer)
            print(answer)

            #--------------------------------------#
            # Download Images                      #
            #--------------------------------------#
            utils.list_blobs_download(blob_service_client, container_name, storagepath)
            st.image(os.path.join("downloads", storagepath))

            #--------------------------------------#
            # Prepare for the next iteration       #
            #--------------------------------------#
            utils.append_conversation_history(messages, response, role="assistant")

            st.session_state['messages'] = messages
            #st.session_state['image_urls'] = image_urls
            st.session_state['storagepath'] = storagepath
            
        else:
            messages = st.session_state['messages']
            #image_urls = st.session_state['image_urls']
            storagepath = st.session_state['storagepath']

            content = []

            content.append({"type": "text", "text": query})
            messages.append({"role": "user","content":content})

            st.write("Retrieving answer from GPT-4 with Vision")
            print("Retrieving answer from GPT-4 with Vision")

            response = utils.gpt4v_query(messages, aoai_endpoint, aoai_api_key, api_version)
            answer = response.json()["choices"][0]["message"]["content"]
            st.write(answer)
            print(answer)

            #for url in image_urls:
            #    st.image(os.path.join("downloads", url['title']))
            st.image(os.path.join("downloads", storagepath))

            utils.append_conversation_history(messages, response, role="assistant")
            st.session_state['messages'] = messages


if __name__ == '__main__':
    main()

