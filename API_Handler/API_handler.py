from datetime import datetime
import hug
import json
import os
from hug.middleware import CORSMiddleware

def get_users_list():
    with open('./users/users.json') as json_file:
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
                os.makedirs('./users/' + user_name)
                # Add user to database
                user_list.append(user_name)
                data = {}
                data["user_list"] = user_list
                with open('users/users.json', 'w') as outfile:
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
                with open('users/users.json', 'w') as outfile:
                    json.dump(data, outfile)

                # Delete user directory
                os.rmdir('./users/' + user_name)

            return json.dumps({'message': 'User deleted'})
    except Exception as e:
        return json.dumps({'error': str(e)})

@hug.post('/getUsers', output=hug.output_format.json)
def getUsers():
    try:
        user_list = get_users_list()
        return json.dumps({'user_list': user_list})
    except Exception as e:
        return json.dumps({'error': str(e)})

@hug.post('/addFaceData')
def upload_file(image_data: "image", user_name: hug.types.text):
    try:
        #image_data = body["image"]
        #user_name = body["user_name"]
        user_list = get_users_list()
        if user_name not in user_list:
            return json.dumps({'error': 'User does not exist'})
        else:
            if image_data:
                date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
                # Spécifiez le chemin du fichier où vous souhaitez enregistrer l'image
                file_path = './' + user_name + '/' + date_str + '.png'  # Modifiez selon vos besoins

                # Enregistrez l'image dans un fichier
                with open(file_path, 'wb') as image_file:
                    image_file.write(image_data)

                # Vous pouvez retourner une réponse JSON pour indiquer le succès
                return json.dumps({'message': 'Image enregistree avec succes'})
            else:
                return json.dumps({'error': 'Aucune image telechargee'})
    except Exception as e:
        return json.dumps({'error': str(e)})

@hug.post('/recognizeFace')
def recognizeFace(image_data = "image"):
    try:
        if image_data:
            # Spécifiez le chemin du fichier où vous souhaitez enregistrer l'image
            file_path = './temp.png'  # Modifiez selon vos besoins

            # Enregistrez l'image dans un fichier
            with open(file_path, 'wb') as image_file:
                image_file.write(image_data)

            #Fonction de reconnaissance
            #TODO en attente de la fonction de reconnaissance on demande d'entrée dans la consol
            res = input("User_name: ")


            res = "User_name res"
            return json.dumps({'user_name': res})

    except Exception as e:
        return json.dumps({'error': str(e)})

# Initialize your hug API
api = hug.API(__name__)

# Enable CORS support
cors_middleware = CORSMiddleware(api, allow_origins=["*"])

# Apply the CORS middleware to your API
api.http.add_middleware(cors_middleware)

# ... Your existing code for API routes ...

def start():
    # Run your API on port 8000
    api.http.serve(port=8400)