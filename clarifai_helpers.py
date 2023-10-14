from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
import streamlit as st
from PIL import Image
import io

class ClarifaiModel:
    def __init__(self, acc_token, user_id, app_id, model_id):
        channel = ClarifaiChannel.get_grpc_channel()
        self.stub = service_pb2_grpc.V2Stub(channel)
        self.metadata = (('authorization', 'Key ' + acc_token),)
        self.userDataObject = resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id)
        self.model_id = model_id

    def convert_img_to_bytes(self, image):
        image_stream = io.BytesIO()
        image.save(image_stream, format="JPEG")
        image_bytes = image_stream.getvalue()
        image_stream.close()
        return image_bytes

    def run(self, image):
        img_bytes = self.convert_img_to_bytes(image)
        post_model_outputs_response = self.stub.PostModelOutputs(
            service_pb2.PostModelOutputsRequest(
                user_app_id=self.userDataObject,
                model_id=self.model_id,
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(
                            image=resources_pb2.Image(
                                base64=img_bytes
                            )
                        )
                    )
                ]
            ),
            metadata=self.metadata
        )
        if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
            print(post_model_outputs_response.status)
            raise Exception("Post model outputs failed, status: " + post_model_outputs_response.status.description)

        return post_model_outputs_response.outputs[0]

if __name__ == '__main__':
    # Image Captioning
    image_captioner = ClarifaiModel(st.secrets['clarifai_key'], 'salesforce', 'blip', 'general-english-image-caption-blip-2')
    image = Image.open('images/scene.jpeg')
    response = image_captioner.run(image)
    print(response.data.text.raw)

    # OCR
    ocr = ClarifaiModel(st.secrets['clarifai_key'], 'clarifai', 'main', 'ocr-scene-english-paddleocr')
    image = Image.open('images/board.jpg')
    response = ocr.run(image)
    text = []
    for region in response.data.regions:
        text.append(region.data.text.raw)
    print(' '.join(text))

    # Color Recognition
    color_recognizer = ClarifaiModel(st.secrets['clarifai_key'], 'clarifai', 'main', 'color-recognition')
    image = Image.open('images/color.jpg')
    response = color_recognizer.run(image)
    print(response.data.colors[0].w3c.name)
