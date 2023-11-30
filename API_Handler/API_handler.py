from datetime import datetime
import hug
import json
import os
from hug.middleware import CORSMiddleware

import fonctionunique
from fonctionunique import *
from base64 import b64decode, b64encode
from io import BytesIO
from PIL import Image

DB_PATH = "./users/"
def get_users_list():
    with open(DB_PATH + "users.json") as json_file:
        data = json.load(json_file)
        return data["user_list"]



@hug.post('/enrollUser')
# add user to database
def enrollUser(user_name: hug.types.text):
    try:
        if user_name:
            user_list = get_users_list()
            if user_name in user_list:
                return json.dumps({'error': 'User already exists'})
            else:
                # MakeAdirectory
                os.makedirs(DB_PATH + user_name)
                # Add user to database
                user_list.append(user_name)
                data = {}
                data["user_list"] = user_list
                with open(DB_PATH + '/users.json', 'w') as outfile:
                    json.dump(data, outfile)

            return json.dumps({'message': 'User added'})
    except Exception as e:
        return json.dumps({'error': str(e)})

@hug.post('/deleteUser')
def deleteUser(user_name: hug.types.text):
    try:
        if user_name:
            user_list = get_users_list()
            if user_name not in user_list:
                return json.dumps({'error': 'User does not exist'})
            else:
                # Remove user from database
                user_list.remove(user_name)
                data = {}
                data["user_list"] = user_list
                with open('./users/users.json', 'w') as outfile:
                    json.dump(data, outfile)

                # Delete user directory
                os.rmdir('./users/' + user_name)

            return json.dumps({'message': 'User deleted'})
    except Exception as e:
        return json.dumps({'error': str(e)})

@hug.post('/getUsers', output=hug.output_format.json)
def getUsers():
    print("getUsers")
    try:
        user_list = get_users_list()
        print(json.dumps({'user_list': user_list}))
        return json.dumps({'user_list': user_list})
    except Exception as e:
        print(json.dumps({'error': str(e)}))
        return json.dumps({'error': str(e)})

def convertImgRawToBase64(imgRaw):
    return b64encode(imgRaw)

def saveImg(file_path, rawimage):
    #check if the image is base64 encoded
    img = rawimage
    try:
        rawimage.startswith(b'b\'')
    except:
        img = b64decode(rawimage)

    # Enregistrez l'image dans un fichier
    with open(file_path, 'wb') as image_file:
        image_file.write(img)

@hug.post('/addFaceData')
def upload_file(image_data: "image", user_name: hug.types.text):
    try:
        #image_data = body["image"]
        #user_name = body["user_name"]
        print(user_name)
        user_list = get_users_list()
        if user_name not in user_list:
            print('User does not exist')
            return json.dumps({'error': 'User does not exist'})
        else:
            print('User found')
            if image_data:
                print('Image telechargee')
                date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
                print(date_str)
                # Spécifiez le chemin du fichier où vous souhaitez enregistrer l'image
                file_path = "temp_add.jpg"  # Modifiez selon vos besoins
                print(file_path)
                saveImg(file_path, image_data)

                img = cv2.imread(file_path)
                img = fonctionunique.crop(img)
                file_path = DB_PATH + user_name + '/' + date_str + '.jpg'  # Modifiez selon vos besoins
                img = fonctionunique.redim(img, 300)
                cv2.imwrite(file_path, img)

                return json.dumps({'message': 'Image enregistree avec succes'})
            else:
                print('Aucune image telechargee')
                return json.dumps({'error': 'Aucune image telechargee'})
    except Exception as e:
        print(json.dumps({'error': str(e)}))
        return json.dumps({'error': str(e)})

@hug.post('/recognizeFace')
def recognizeFace(image_data = "image"):
    print("recognizeFace")
    try:
        if image_data:
            # Spécifiez le chemin du fichier où vous souhaitez enregistrer l'image
            file_path = DB_PATH + 'temp.png'  # Modifiez selon vos besoins

            # Enregistrez l'image dans un fichier
            saveImg(file_path, image_data)

            res = fonctionunique.reconnaissace_visage(file_path)
            json_res = json.dumps({'user_name': res})
            print(json_res)
            return json.dumps({'user_name': res})

    except Exception as e:
        print(json.dumps({'error': str(e)}))
        return json.dumps({'error': str(e)})

def start(ip):
    #intit face recognition
    user_list = get_users_list()
    for user in user_list:
        fonctionunique.addUser(user)

    print(fonctionunique.num_to_name)
    # Initialize your hug API
    api = hug.API(__name__)

    # Enable CORS support
    cors_middleware = CORSMiddleware(api, allow_origins=["*"])

    # Apply the CORS middleware to your API
    api.http.add_middleware(cors_middleware)
    # Run your API on port 8000
    api.http.serve(port=8400, host=ip)