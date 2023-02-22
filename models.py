from attr import fields_dict
from flask import Flask, jsonify, request, session, redirect, flash, url_for
from passlib.hash import pbkdf2_sha256
from bson.objectid import ObjectId
import sys
import app
import uuid
import os

# Class User, called from app.py
class User:

    # Session begins when the user logs in
    def start_session(self, user):
        del user['password']
        session['logged_in'] = True
        session['user'] = user
        session['group'] = {}
        return jsonify(user), 200

    # Session begins when the admin logs in
    def start_admin_session(self, admin):
        del admin['password']
        session['adminlogged_in'] = True
        session['admin'] = admin
        return jsonify(admin), 200

    # function to check the password
    def password_check(self, passwd):

        SpecialSym = ['$', '@', '#', '%', '_', '-']
        val = True

        if len(passwd) < 6:
            print('length should be at least 6')
            val = False

        if len(passwd) > 20:
            print('length should be not be greater than 8')
            val = False

        if not any(char.isdigit() for char in passwd):
            print('Password should have at least one numeral')
            val = False

        if not any(char.isupper() for char in passwd):
            print('Password should have at least one uppercase letter')
            val = False

        if not any(char.islower() for char in passwd):
            print('Password should have at least one lowercase letter')
            val = False

        if not any(char in SpecialSym for char in passwd):
            print('Password should have at least one of the special characters')
            val = False
        if val:
            return val

    # function to sign up the user by admin
    def signup(self):
        # Create the user object
        user = {
            "_id": uuid.uuid4().hex,
            "name": request.form.get('name'),
            "email": request.form.get('email'),
            "password": request.form.get('password')
        }

        password = user['password']
        if (self.password_check(password)):
            user['password'] = pbkdf2_sha256.encrypt(user['password'])
        else:
            return jsonify({"error": "Password should be greater than 6 in size, with one capital letter and a special symbol"}), 400

        # Encrypt the password
        # user['password'] = pbkdf2_sha256.encrypt(user['password'])

        # Check for existing email address
        if app.db.users.find_one({"email": user['email']}):
            return jsonify({"error": "Email address already in use"}), 400

        if app.db.users.insert_one(user):
            flash('User Sign Up Successful')
            return jsonify(session['admin']['username']), 200

        return jsonify({"error": "Signup failed"}), 400

    # function to signout which clears session
    def signout(self):
        session.clear()
        return redirect('/')

    # login function
    def login(self):
        user = app.db.users.find_one({
            "email": request.form.get('email')
        })
        if user and pbkdf2_sha256.verify(request.form.get('password'), user['password']):
            return self.start_session(user)

        return jsonify({"error": "Invalid login credentials"}), 401

    # function to change the password
    def changepassword(self):
        user = session['user']
        print(user)
        email = user['email']

        user1 = {
            "_id": user["_id"],
            "name": user['name'],
            "email": email,
            "password": request.form.get('password'),
            "confirmpassword": request.form.get('confirmpassword'),
        }

        if user1["password"] != user1["confirmpassword"]:
            return jsonify({"error": "Passwords not matching"}), 401
        else:
            del user1['confirmpassword']

        password = user1['password']

        if (self.password_check(password)):
            user1['password'] = pbkdf2_sha256.encrypt(user1['password'])
        else:
            return jsonify({"error": "Password should be greater than 6 in size, with one capital letter and a special symbol"}), 400

        # user['password'] = pbkdf2_sha256.encrypt(user['password'])

        if app.db.users.find_one({"email": email}):
            app.db.users.update_one(
                {'email': email}, {"$set": {'password': user1['password']}}, upsert=False)
            flash('Password Change Successful')
            return self.start_session(user1)

        return jsonify({"error": "User Not found, please sign up"}), 401

    # function to change the password of the users by admin
    def chpassbyadmin(self):
        user_ = request.form.get('username')
        user = app.db.users.find_one({"name": user_})

        user1 = {
            "_id": user["_id"],
            "name": user['name'],
            "email": user['email'],
            "password": request.form.get('password'),
            "confirmpassword": request.form.get('confirmpassword'),
        }

        if user1["password"] != user1["confirmpassword"]:
            return jsonify({"error": "Passwords not matching"}), 401
        else:
            del user1['confirmpassword']

        password = user1['password']
        if (self.password_check(password)):
            user1['password'] = pbkdf2_sha256.encrypt(user1['password'])
        else:
            return jsonify({"error": "Password should be greater than 6 in size, with one capital letter and a special symbol"}), 400

        # # user['password'] = pbkdf2_sha256.encrypt(user['password'])

        if app.db.users.find_one({"email": user1['email']}):
            print("Changed")
            app.db.users.update_one({'email': user1['email']}, {
                                    "$set": {'password': user1['password']}}, upsert=False)
            flash('Password Change Successful')
            return jsonify(session['admin']['username']), 200

        return jsonify({"error": "User Not found, please sign up"}), 401

    # removing the user
    def removeUser(self, uname):
        user = app.db.users.find_one({"name": uname})
        user1 = {
            "_id": user["_id"],
            "name": user['name'],
            "email": user['email'],
        }
        app.db.users.delete_one({"_id": user1["_id"]})

        return redirect('/userlist')

    # deletes a file from the database
    def deletefile(self, fid, uid, uname):

        # deletes an item from fs.files
        app.admindb.get_collection('fs.files').delete_one({
            '_id': ObjectId(fid)})

        # deletes all items related to the file in fs.chunks
        app.admindb.get_collection('fs.chunks').delete_many(
            {'files_id': ObjectId(fid)})

        chunks = app.admindb.get_collection(
            'fs.files').find({'original_file_id': fid})

        for i in chunks:
            app.admindb.get_collection(
                'fs.files').delete_one({'_id': i['_id']})
            app.admindb.get_collection('fs.chunks').delete_many({
                'files_id': i['_id']})
        return redirect(url_for('showFiles', name=uname, uid=uid))

    # deletes all file from the database
    def deleteAll(self):
        app.admindb.get_collection('fs.chunks').delete_many({})
        app.admindb.get_collection('fs.files').delete_many({})

        return redirect(url_for('showFiles', name=session['user']['name'], uid=session['user']['_id']))

    # showing files
    def showFiles(self, user_, archived):
        users = app.db.users.find_one({'name': user_})
        files = app.admindb.get_collection('fs.files').find(
            {"username": users['name'], "archived": archived})

        return files

    # showing files by admin
    def showFilesbyAdmin(self, user_):
        group_ids = {}
        users = app.db.users.find_one({'name': user_})
        files = app.admindb.get_collection(
            'fs.files').find({"username": users['name']})
        groups = app.admindb.get_collection('fs.files').aggregate([{"$match": {"username": users['name']}},
                                                                   {'$group': {
                                                                       '_id': {"grp": "$group"},
                                                                       "file_ids": {
                                                                           "$addToSet": "$_id"
                                                                       }}
                                                                    }])

        for i, g in enumerate(groups):
            group_ids[g['_id']['grp']] = i+1

        return files, group_ids

    # admin login function
    def adminlogin(self):
        username = "admin"
        password = "admin"
        admin = {
            'username': request.form.get('username'),
            'password': request.form.get('password')
        }

        if admin['username'] == username and admin['password'] == password:
            return self.start_admin_session(admin)
        return jsonify({"error": "Invalid login credentials"}), 401

    # retrieves all user from the database on admin page
    def userlist(self):
        users = app.db.users.find()
        return users

    # function to retrieve all files of a user
    def getallfiles(self):
        curr_user = session['user']['name']
        user_file_path = './static/user_files/'+curr_user
        files_dict = {}
        print(session)
        for folder_name in os.listdir(user_file_path):
            if folder_name != '.DS_Store' and not folder_name in files_dict:
                files_arr = {}
                i = 0
                for file in os.listdir(user_file_path+"/"+folder_name):
                    if not file.startswith("HighLighted++") and not file == "tables" and file != '.DS_Store':
                        files_arr["file"+str(i)] = file
                        files_arr["result"+str(i)] = "HighLighted++"+file
                        i += 1
                files_dict[folder_name] = files_arr

        return files_dict
