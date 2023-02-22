
from email.mime import base
from fileinput import filename
from genericpath import exists
from string import hexdigits
from flask import Flask, request, redirect, jsonify, render_template, session, url_for, Response, flash, send_file
import os
import time
import shutil
from pygments import highlight
from flask_dropzone import Dropzone
# from pdf2text_tika import *
from flask_pymongo import PyMongo, MongoClient
from gridfs import Collection, GridFS
import random
from functools import wraps
from models import *
from PyPDF2 import PdfFileMerger, PdfFileReader
from os import listdir
from os.path import isfile, join
from Recomendation_system_v11 import *
import hashlib
from pathlib import Path
from flask import Flask, send_from_directory
from visualization import *
import jyserver.Flask as jsf
from flask_wtf.csrf import CSRFProtect
from PIL import Image
import io
import codecs
import base64
from warnings import filterwarnings
import uuid
import zipfile
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.int` is a deprecated alias')
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.object` is a deprecated alias')
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.float` is a deprecated alias')


app = Flask(__name__)
dropzone = Dropzone(app)
app.secret_key = b'\xcc^\x91\xea\x17-\xd0W\x03\xa7\xf8J0\xac8\xc5'
csrf = CSRFProtect(app)


# Database
connectionString = "mongodb://127.0.0.1:27017/admin"


app.config["MONGO_URI"] = connectionString  # "mongodb://localhost:27017/1023"
MONGO_CLIENT = MongoClient('mongodb://127.0.0.1:27017/')
db = MONGO_CLIENT['user_login_system']
admindb = MONGO_CLIENT['admin']

GRID_FS = GridFS(admindb)

mongo = PyMongo(app)
app.config["TIKA_LOG_FILE"] = ""

# Decorators


def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            return redirect('/')
    return wrap


def admin_login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'adminlogged_in' in session:
            return f(*args, **kwargs)
        else:
            return redirect('/admin')
    return wrap

# decorator to check if either admin or user is logged in


def is_granted(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if ('logged_in' in session) or ('adminlogged_in' in session):
            return f(*args, **kwargs)
        else:
            return redirect('/')
    return wrap


basedir = os.path.abspath(os.path.dirname(__file__))

# app.config.update(
#     # UPLOADED_PATH=os.path.join(basedir, filefolder),
#     # Flask-Dropzone config:
#     DROPZONE_MAX_FILE_SIZE=10,
#     DROPZONE_MAX_FILES=50,
#     DROPZONE_UPLOAD_ON_CLICK=True,
# )
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = '.pdf'

# admin login page is returned from this call, triggerd when "Go to Admin Page" is clicked on login.html


@app.route('/admin')
def admin():
    return render_template('adminlogin.html')

# called when the admin is logged in


@app.route('/user/adminlogin', methods=['POST'])
def adminlogin():
    return User().adminlogin()

# user signup page for admin


@app.route('/user/signup', methods=['POST'])
@admin_login_required
def signup():
    return User().signup()


@app.route('/user/changepassword', methods=['POST'])
def changepassword():
    return User().changepassword()

# lets the user change the password


@app.route("/changePasswordUser")
@login_required
def changePasswordUser():
    return render_template("changepassword.html")


# Change the password by admin- open
@app.route('/chpassbyadmin/', methods=['POST'])
@admin_login_required
def chpassbyadmin():
    return User().chpassbyadmin()

# lets the admin change the passowrd of user


@app.route("/changePassbyAdmin/<name>")
@admin_login_required
def changePassbyAdmin(name):
    return render_template("chpassbyadmin.html", uname=name)

# Chnage the password by admin- close

# remove user by admin


@app.route("/removeUser/<name>")
@admin_login_required
def removeUser(name):
    return User().removeUser(name)

# Section that returns the visualizations to view when the "View Results" link is clicked. This is retrieved from the database
# Returns the highlighted document


@app.route('/visual/')
@is_granted
def visuals():
    file = admindb.get_collection('fs.files').find_one(
        {'original_file_id': request.args['fid'], 'filename': request.args['fname']})
    pdffile = GRID_FS.get(file_id=file['_id'])
    base64_data = codecs.encode(pdffile.read(), 'base64')
    pdffile = base64_data.decode('utf-8')
    return pdffile

# returns the "similarity chart between documents. 3D if multiple documents else 2D"


@app.route('/visual1/')
@is_granted
def visuals1():
    image = admindb.get_collection('fs.files').find_one(
        {'original_file_id': request.args['fid'], 'filename': request.args['fname']})

    overall_sim = GRID_FS.get(file_id=image['_id'])
    base64_data = codecs.encode(overall_sim.read(), 'base64')
    overall_sim = base64_data.decode('utf-8')
    return overall_sim

# returns class similarity


@app.route('/visual2/')
@is_granted
def visuals2():
    image = admindb.get_collection('fs.files').find_one(
        {'original_file_id': request.args['fid'], 'filename': request.args['fname']})

    overall_sim = GRID_FS.get(file_id=image['_id'])
    base64_data = codecs.encode(overall_sim.read(), 'base64')
    overall_sim = base64_data.decode('utf-8')
    return overall_sim


# show files by admin
@app.route('/showFiles/<name>/<uid>')
# @admin_login_required
def showFiles(name, uid):
    usersfiles, group_ids = User().showFilesbyAdmin(name)
    return render_template("showfiles.html", user_files=usersfiles, username=name, user_id=uid, group_ids=group_ids)


@app.route('/file/<fname>')
@login_required
def files(fname):
    return mongo.send_file(fname)


@app.route('/result/<fid>')
@login_required
def result(fid):
    result_file = admindb.get_collection(
        "rs.files").find_one({'original_file_id': fid})
    return mongo.send_file(result_file['filename'], base="rs")

# uploads the results from "Results" section on homepage. This link is used only if the user unchecks the "Save Results" while uploading the document/s


@app.route('/upload_results/<folder>/<fname>')
@login_required
def upload_results(folder, fname):
    curr_user_id = session['user']['_id']
    curr_user = session['user']['name']
    user_file_path = './static/user_files/'+curr_user+'/'+folder
    highlighted_file = "HighLighted++"+fname
    original_file = admindb.get_collection("fs.files").find_one(
        {"filename": fname, "userid": curr_user_id, "results": False})

    fobj = open(user_file_path+'/'+highlighted_file, 'rb')
    mongo.save_file(highlighted_file, fobj, base="fs",
                    original_file_id=str(original_file['_id']), archived=False)
    fobj.close()

    visual_files = [v for v in os.listdir(user_file_path+"/tables/")]

    for img in visual_files:
        if img.startswith(fname) and img.endswith(".png"):
            img_contents = open(user_file_path+'/tables/'+img, 'rb').read()
            img_name = img.replace(fname+'_', "")
            GRID_FS.put(img_contents, filename=img_name,
                        original_file_id=str(original_file['_id']))

    img_contents = open(user_file_path+'/tables/overall_sim.png', 'rb').read()

    GRID_FS.put(img_contents, filename="overall_sim.png",
                original_file_id=str(original_file['_id']))

    if 'results_saved' in session:
        temp_list = session['results_saved']
        temp_list.append(fname)
        session['results_saved'] = temp_list
    else:
        session['results_saved'] = [fname]

    myquery = {"filename": fname, "userid": curr_user_id, "results": False}
    newvalues = {"$set": {"results": True}}
    admindb.get_collection('fs.files').update_one(myquery, newvalues)
    return redirect(url_for('home'))

# uploads the files from "Results" section on homepage. This link is used only if the user unchecks the "Save Files" while uploading the document/s


@app.route('/upload_files/<folder>/<fname>')
@login_required
def upload_files(folder, fname):
    curr_user = session['user']
    parent_dir = "./static/user_files/"
    userfolder = curr_user['name']
    userpath = os.path.join(parent_dir, userfolder)
    session['userfolder'] = userfolder
    user_file_path = './static/user_files/'+userfolder+'/'+folder
    unique_grp_id = session['group'][folder]
    if 'results_saved' in session:
        if fname in session['results_saved']:
            res = True
        else:
            res = False
    else:
        res = False

    fobj = open(user_file_path+'/'+fname, 'rb')
    mongo.save_file(fname, fobj, base='fs',
                    userid=curr_user['_id'], username=curr_user['name'], results=res, archived=False, group=str(unique_grp_id))
    fobj.close()
    if 'files_saved' in session:
        session['files_saved'].append(fname)
    else:
        session['files_saved'] = [fname]

    return redirect(url_for('home'))

# deletes a file from the database


@app.route('/deletefile/<fid>/<uid>/<uname>')
@is_granted
def deletefile(fid, uid, uname):
    return User().deletefile(fid, uid, uname)

# deletes all file from the database


@app.route('/deleteAll/')
def deleteAll():
    return User().deleteAll()

# user signout


@app.route('/user/signout')
def signout():
    filefolder = session.get('userfolder')
    if filefolder == None:
        return User().signout()
    parent_dir = './static/user_files/'
    shutil.rmtree(filefolder, ignore_errors=True)
    shutil.rmtree(parent_dir+filefolder, ignore_errors=True)
    session.pop('userfolder', None)
    session.pop('curr_file', None)
    print("folder removed successfully")
    return User().signout()

# user login


@app.route('/user/login', methods=['POST'])
def login():
    return User().login()

# go back button


@app.route('/goback')
def goback():
    if 'admin' in session:
        return redirect(url_for("users"))
    elif 'user' in session:
        return redirect(url_for("home"))
    else:
        return redirect(url_for('/'))


@app.route("/")
def loginUser():
    return render_template("login.html")

# user signup by admin


@app.route("/signupUser/")
@admin_login_required
def signupUser():
    return render_template("signup.html")

# displays all signed up user on admin's page


@app.route("/userlist")
@admin_login_required
def users():
    users_ = User().userlist()
    return render_template("userlist.html", users=users_)

# this displays a restricted access page
# works when the user copy and paste the homepage url into another tab in the same browser


@app.route("/restricted/")
def error_page():
    return render_template("error.html")

# homepage


@app.route("/home/")
@login_required
def home():
    curr_user = session['user']['name']
    group_ids = {}
    for i, g in enumerate(session['group'].values()):
        group_ids[g] = i+1

    if 'userfolder' in session:

        files_dict = User().getallfiles()
        user_file_path = './static/user_files/'+curr_user
        return render_template("home.html", path=user_file_path, folders=files_dict, res=True, group_ids=group_ids)
    else:
        return render_template("home.html", res=False)

# uploads documents into the local storage with an option to save it to the database ("Save Files", "Save Results")
# called when the dropzone form in submitted from homepage


@app.route('/upload', methods=['POST', 'GET'])
@login_required
def uploadpdf():
    curr_user = session['user']
    parent_dir = "./static/user_files/"
    userfolder = curr_user['name']
    userpath = os.path.join(parent_dir, userfolder)
    session['userfolder'] = userfolder
    session['save_file'] = request.form.get('storeFiles') == 'files'
    session['save_result'] = request.form.get('storeResult') == 'results'
    uploaded_files = []

    # a unique group id to track the documents that were uploaded together
    unique_grp_id = uuid.uuid1()
    filefolder = str(random.randint(1000000000, 9999999999))

    session['group'][filefolder] = str(unique_grp_id)

    if request.method == "POST":
        for key, f in request.files.items():

            # if the save files is checked then it will be saved in local system
            if session['save_file']:
                if 'files_saved' in session:
                    session['files_saved'].append(f.filename)
                else:
                    session['files_saved'] = [f.filename]

                if key.startswith('file'):
                    mongo.save_file(f.filename, f, base='fs', userid=curr_user['_id'], username=curr_user['name'], results=session['save_result'], archived=False, group=str(unique_grp_id))

            # creates a folder in local storage to save the files and results
            if not os.path.exists(userpath):
                os.makedirs(userpath)
                print("Directory ", userpath,  " Created ")
            else:
                print("Directory ", userpath,  " already exists")

            userfiles = os.path.join(userpath, filefolder)

            if not os.path.exists(userfiles):
                os.makedirs(userfiles)
            uploaded_files.append(f.filename)

            f.seek(0)
            f.save(os.path.join(userfiles, f.filename))
        session['curr_file'] = [filefolder, uploaded_files]

    return render_template('output.html')

# this is called when the "Submit" button is clicked on the homepage after uploading the documents


@app.route('/output_page', methods=['POST', 'GET'])
@login_required
def output_page():
    curr_user = session['user']['name']
    curr_user_id = session['user']['_id']
    filefolder = session['curr_file'][0]
    file_name = session['curr_file'][1]
    user_file_path = './static/user_files/'+curr_user+'/'+filefolder
    user_path = "./static/user_files/"+curr_user
    # This is the function that processes all documents in the folder and saves the result in local storage.
    # If the user has checked "Save Results" while uploading documents, then the result is saved in the database too
    # This below function is present in "Recommendation_system_v11.py"
    extractAll(user_file_path)

    # this function is used to create visualization using the CSV file created by the above function
    visualize(user_file_path+"/tables")

    files = [f for f in os.listdir(user_file_path)]

    visual_files = [v for v in os.listdir(user_file_path+"/tables/")]
    for fname in file_name:
        if 'processed' in session:
            session['processed'] += [fname]
        else:
            session['processed'] = [fname]
        if "HighLighted++"+fname in files:
            highlighted_file = "HighLighted++"+fname
            if session['save_result']:
                if 'results_saved' in session:
                    session['results_saved'].append(fname)
                else:
                    session['results_saved'] = [fname]

                original_file = admindb.get_collection("fs.files").find_one(
                    {"filename": fname, "userid": curr_user_id})
                try:
                    og_file_id = str(original_file['_id'])
                except:
                    og_file_id = None
                fobj = open(user_file_path+'/'+highlighted_file, 'rb')
                mongo.save_file(highlighted_file, fobj, base="fs",
                                original_file_id=og_file_id, archived=False)

                for img in visual_files:
                    if img.startswith(fname) and img.endswith(".png"):
                        img_contents = open(
                            user_file_path+'/tables/'+img, 'rb').read()
                        img_name = img.replace(fname+'_', "")
                        GRID_FS.put(img_contents, filename=img_name,
                                    original_file_id=og_file_id, archived=False)
                img_contents = open(
                    user_file_path+'/tables/overall_sim.png', 'rb').read()
                GRID_FS.put(img_contents, filename="overall_sim.png",
                            original_file_id=og_file_id, archived=False)
                fobj.close()
    return redirect(url_for('home'))

# Processes the file from "Result" section on homepage.
# used when the user only uploads the documents without submitting it for processing
# works same way as the above function


@app.route('/process_file/<filefolder>/<file_name>', methods=['POST', 'GET'])
@login_required
def process_file(filefolder, file_name):
    curr_user = session['user']['name']
    curr_user_id = session['user']['_id']
    filefolder = filefolder
    user_file_path = './static/user_files/'+curr_user+'/'+filefolder
    file_name = [f for f in os.listdir(user_file_path)]

    extractAll(user_file_path)
    visualize(user_file_path+"/tables")

    files = [f for f in os.listdir(user_file_path)]

    visual_files = [v for v in os.listdir(user_file_path+"/tables/")]
    for fname in file_name:
        if 'processed' in session.keys():

            session['processed'] += [fname]
        else:
            session['processed'] = [fname]
        if "HighLighted++"+fname in files:
            highlighted_file = "HighLighted++"+fname
            if session['save_result']:
                if 'results_saved' in session:
                    session['results_saved'].append(fname)
                else:
                    session['results_saved'] = [fname]

                original_file = admindb.get_collection("fs.files").find_one(
                    {"filename": fname, "userid": curr_user_id})
                try:
                    og_file_id = str(original_file['_id'])
                except:
                    og_file_id = None
                fobj = open(user_file_path+'/'+highlighted_file, 'rb')
                mongo.save_file(highlighted_file, fobj, base="fs",
                                original_file_id=og_file_id, archived=False)

                for img in visual_files:

                    if img.startswith(fname) and img.endswith(".png"):
                        img_contents = open(
                            user_file_path+'/tables/'+img, 'rb').read()
                        img_name = img.replace(fname+'_', "")
                        GRID_FS.put(img_contents, filename=img_name,
                                    original_file_id=og_file_id, archived=False)
                img_contents = open(
                    user_file_path+'/tables/overall_sim.png', 'rb').read()
                GRID_FS.put(img_contents, filename="overall_sim.png",
                            original_file_id=og_file_id, archived=False)
                fobj.close()
    return redirect(url_for('home'))


def remove_directory():
    filefolder = session.get('filefolder', None)
    shutil.rmtree(filefolder, ignore_errors=True)
    session.pop('filefolder', None)
    print("folder removed successfully")

# triggered when "Clear Session" link is clicked on homepage or if the user signs out


@app.route('/end_session', methods=['POST', 'GET'])
@is_granted
def end_session():
    filefolder = session.get('userfolder')
    if filefolder == None:
        return redirect(url_for('home'))
    parent_dir = './static/user_files/'
    shutil.rmtree(filefolder, ignore_errors=True)
    shutil.rmtree(parent_dir+filefolder, ignore_errors=True)
    session.pop('userfolder', None)
    session.pop('curr_file', None)
    session.pop('processed', None)
    session.pop('files_saved', None)
    session.pop('results_saved', None)
    session['group'] = {}
    print("folder removed successfully")
    print("AFTER          ", session)
    return redirect(url_for('home'))

# downloads the results in the zip format (homepage: from loacl storage)


@app.route('/download_zip/<fname>/<folder>')
@login_required
def download_zip(fname, folder):

    curr_user = session['user']
    parent_dir = "./static/user_files/"
    userfolder = curr_user['name']
    userpath = os.path.join(parent_dir, userfolder)
    session['userfolder'] = userfolder
    user_file_path = './static/user_files/'+userfolder+'/'+folder
    original_filenames = []

    for og_file in os.listdir(user_file_path):
        if not og_file.startswith("HighLighted++") and og_file.endswith(".pdf"):
            original_filenames.append(og_file[:-4])

    temp_folder = "temp"

    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)

    for y in original_filenames:
        highlighted_file = "HighLighted++"+y+".pdf"
        if not os.path.exists(temp_folder+'/'+y):
            os.mkdir(temp_folder+'/'+y)

        for x in os.listdir(user_file_path+'/tables/'):
            if x.startswith(y) and x.endswith('.png'):
                img_name = x.replace(y+".pdf_", "")
                shutil.copy(user_file_path+'/tables/'+x,
                            temp_folder+'/'+y+'/'+img_name)
        shutil.copy(user_file_path+'/'+highlighted_file,
                    temp_folder+'/'+y+'/'+highlighted_file)
        shutil.copy(user_file_path+'/tables/overall_sim.png',
                    temp_folder+'/'+y+'/overall_sim.png')
    shutil.make_archive(os.path.expanduser(
        "~")+'/Downloads/results', 'zip', temp_folder)
    shutil.rmtree(temp_folder)
    return redirect(url_for('home'))

# downloads the results from database


@app.route('/download_from_db/<file_id>/<filename>/<img>/<uname>/<uid>')
@is_granted
def download_from_db(file_id, filename, img, uname, uid):
    res_files = admindb.get_collection(
        'fs.files').find({'original_file_id': file_id})
    res_file_ids = {str(file['_id']): file['filename'] for file in res_files}
    temp_folder = "temp"
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)

    for file_id, file_name in res_file_ids.items():
        file_content = GRID_FS.get(file_id=ObjectId(file_id)).read()
        output = open(temp_folder+'/'+file_name, 'wb')
        output.write(file_content)
        output.close()
    shutil.make_archive(os.path.expanduser(
        "~")+'/Downloads/results', 'zip', temp_folder)
    shutil.rmtree(temp_folder)

    return redirect(url_for('showFiles', name=uname, uid=uid))

# File archive:
# Files are made invisible for the user if he/she decides to archive it, but will be visible to the admin and has the
# ability to unarchive it for the user

# Link to archive files, either by user or the admin


@app.route('/archive_file/<fid>/<uname>/<uid>')
@is_granted
def archive_file(fid, uname, uid):
    myquery1 = {"_id": ObjectId(fid),  "archived": False}
    newvalues1 = {"$set": {"archived": True}}
    admindb.get_collection('fs.files').update_one(myquery1, newvalues1)
    myquery2 = {'original_file_id': fid, 'archived': False}
    newvalues2 = {'$set': {'archived': True}}
    admindb.get_collection('fs.files').update_many(myquery2, newvalues2)
    return redirect(url_for('showFiles', name=uname, uid=uid))

# link to unarchive file, only by the admin


@app.route('/unarchive_file/<fid>/<uname>/<uid>')
@is_granted
def unarchive_file(fid, uname, uid):
    myquery1 = {"_id": ObjectId(fid), "archived": True}
    newvalues1 = {"$set": {"archived": False}}
    admindb.get_collection('fs.files').update_one(myquery1, newvalues1)
    myquery2 = {'original_file_id': fid, 'archived': True}
    newvalues2 = {'$set': {'archived': False}}
    admindb.get_collection('fs.files').update_many(myquery2, newvalues2)
    return redirect(url_for('showFiles', name=uname, uid=uid))


if __name__ == "__main__":
    app.run(port=8000, host='localhost', debug=True, threaded=True)
