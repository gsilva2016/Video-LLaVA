import torch
from videollava.conversation import conv_templates, SeparatorStyle, Conversation
from videollava.serve.gradio_utils import Chat, tos_markdown, learn_more_markdown, title_markdown, block_css
import os
import io
from PIL import Image

from paho.mqtt import client as mqtt_client
broker = 'broker.hivemq.com'
port = 1883
topic = "/roi/#"
client_id = 'server/1'

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc != 0:
            print("Failed to connect, return code %d", rc)

    #client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION1, client_id)
    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(broker, port, 60)
    return client

def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        #save_file_name = msg.topic
        #if "roi-" in save_file_name:
        #    print("Messaged rcved : ", msg.topic, " | ", msg.payload)
        #    return
        
        #save_file_name = save_file_name.split("/")[2]
        #f = open(save_file_name, 'wb')
        #f.write(msg.payload)
        #f.close()
        print ('processing request: ' + msg.topic)

        topicArr = msg.topic.split("/")
        item_id = topicArr[2]
        prompt = topicArr[3].replace("_", " ")

        image_processor = handler.image_processor
        pil_data = Image.open(io.BytesIO(msg.payload)).convert("RGB")
        tensor = image_processor.preprocess(pil_data, return_tensors='pt')['pixel_values'][0]
        tensor = tensor.to(handler.model.device, dtype=dtype)
        images_tensor = []
        images_tensor.append(tensor)

        DEFAULT_IMAGE_TOKEN = "<image>"
        text_en_in = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        print("PROMPT: " + text_en_in)
        state_ = conv_templates[conv_mode].copy()
        text_en_out, state_ = handler.generate(images_tensor, text_en_in, first_run=False, state=state_)
        state_.messages[-1]  = (state_.roles[1], text_en_out)
        text_en_out = text_en_out.split('#')[0]
        textbox_out = text_en_out
        #print("OUTPUT: " + textbox_out)
        topic_out = "/roi-res/" + item_id # + "/" + topicArr[3]
        print("Sending result to ", topic_out)
        client.publish(topic_out, textbox_out)


    client.subscribe(topic)
    #client.subscribe(topic_desc)
    client.on_message = on_message


#### Video-LLaVA ####
conv_mode = "llava_v1"
model_path = 'LanguageBind/Video-LLaVA-7B'
cache_dir = './cache_dir'
device = 'cpu'
load_8bit = False
load_4bit = False # not supported by ITREX
dtype = torch.bfloat16
handler = Chat(model_path, conv_mode=conv_mode, load_8bit=load_8bit, load_4bit=load_8bit, device=device, cache_dir=cache_dir, dtype=dtype)
if not os.path.exists("temp"):
    os.makedirs("temp")

prompt_describe = "describe the item in the image"
prompt_verify_count = "is there one item in the image"



#####################

client = connect_mqtt()
subscribe(client)
print("Server ready...")
client.loop_forever()
