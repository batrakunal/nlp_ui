from __future__ import absolute_import, print_function
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import tweepy
import unicodedata
import time
import threading
import requests
from datetime import datetime
import json
import pandas as pd
import sys
import string


def read_secret_file(file_path):
    # Reads the secret.json file
    if file_path[-4] != "json":
        print("Not a json file")
        return None
    secret_json = file_path
    secrets = json.loads(open(secret_json).read())
    return secrets


class CSVWithBuffer(object):
    def __init__(self, hour_start="07:00:00", hour_end="20:00:00"):
        self.buffer = None
        self.start = datetime.strptime(hour_start, "%H:%M:%S")
        self.end = datetime.strptime(hour_end, "%H:%M:%S")
        self.counter = 1

    def addBuffer(self, pdDataFrame):
        if self.buffer is None:
            self.buffer = pdDataFrame
        else:
            self.buffer = pd.concat([self.buffer, pdDataFrame], axis=0)
        if len(self.buffer) >= 10000:
            self.counter += 1
            t = threading.Thread(
                target=self.create_csv, args=(self.buffer, self.counter)
            )
            t.setDaemon(True)
            t.start()
            self.buffer = []

    def create_csv(self, pdDataFrame, current_count):
        agora = datetime.now()
        if (agora.hour >= self.start.hour & agora.minute >= self.start.minute) | (
            agora.hour < self.end.hour & agora.minute <= self.end.minute
        ):
            start_time = time.time()
            print("Created the CSV " + str(current_count))
            pdDataFrame.to_csv(
                path_or_buf="StreamingCSV_" + str(current_count), index=False
            )
            print("Time elapsed: " + str(time.time() - start_time))
        else:
            print("System in sleep mode...")


class StdOutListener(StreamListener):
    """ A listener handles tweets that are received from the stream.
    This is a basic listener that just prints received tweets to stdout.
    """

    count = 1
    inserter = CSVWithBuffer()

    def on_data(self, data):
        try:
            time_start_get = time.time()
            data_dict = json.loads(data)
            list_to_del = []
            list_to_stay = [
                "text",
                "in_reply_to_status_id",
                "id",
                "favorite_count",
                "retweeted",
                "timestamp_ms",
                "retweet_count",
                "in_reply_to_user_id",
                "user",
                "created_at",
                "place",
                "entities",
            ]
            for k, v in data_dict.iteritems():
                if k not in list_to_stay:
                    list_to_del.append(k)
            for key in list_to_del:
                if key in data_dict.keys():
                    del data_dict[key]
            try:
                status_media_type = str(
                    data_dict["entities"]["media"][0]["type"]
                ).encode("utf-8")
                status_media_url = str(
                    data_dict["entities"]["media"][0]["media_url"]
                ).encode("utf-8")
            except KeyError:
                status_media_type = "null"
                status_media_url = "null"

            data_dict["status_media_type"] = status_media_type
            data_dict["status_media_url"] = status_media_url
            try:
                del data_dict["entities"]
            except KeyError:
                pass
            try:
                data_dict["city_full"] = data_dict["place"]["full_name"]
                data_dict["country"] = data_dict["place"]["country"]
                data_dict["place_type"] = data_dict["place"]["place_type"]
                data_dict["place_id"] = data_dict["place"]["id"]
                data_dict["place_name"] = data_dict["place"]["name"]
            except KeyError:
                data_dict["city_full"] = "null"
                data_dict["country"] = "null"
                data_dict["place_type"] = "null"
                data_dict["place_id"] = "null"
                data_dict["place_name"] = "null"
            try:
                data_dict["user_id"] = data_dict["user"]["id"]
            except KeyError:
                data_dict["user_id"] = "null"
            try:
                del data_dict["place"]
            except KeyError:
                pass
            try:
                insert_obj = pd.DataFrame(data=data_dict, columns=list_to_stay)
                self.inserter.addBuffer(insert_obj)
            except Exception as e:
                print(e)
            return True
        except Exception as e:
            print(e)
            return True
        except:
            print("Problem in the code")
            time.sleep(5)
            return True

    def on_error(self, status):
        print("Entrou no on_error!!!")
        print(status)


def start_tweeter_streaming(secret_file_path, location_coordinates=[], track=[]):
    """
    This function starts a streaming API from tweeter, the function will filter the tweets based on the arguments passed,
    for example, if you pass both track as a list and location_coordinates as a list, it will filter both, if you pass
    only one argument, it will use that one, if you dont pass any, it wont do anything.
    :param secret_file_path:
    :param location_coordinates:
    :param track:
    :return:
    """
    assert isinstance(
        location_coordinates, list
    ), "Location coordinates has to be a list with 4 coordinates"
    try:
        l = StdOutListener()
        secret_json = secret_file_path
        secrets = read_secret_file(secret_json)
        consumer_key = secrets["consumer_key"]
        consumer_secret = secrets["consumer_secret"]
        access_token = secrets["access_key"]
        access_token_secret = secrets["access_token_secret"]
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        # Connect/reconnect the stream
        stream = Stream(auth, l)
        # DON'T run this approach async or you'll just create a ton of streams!
        if len(location_coordinates) != 0 and len(track) == 0:
            stream.filter(locations=location_coordinates)
        elif len(location_coordinates) == 0 and len(track) != 0:
            stream.filter(track=track)
        elif len(location_coordinates) != 0 and len(track) != 0:
            stream.filter(track=track, locations=location_coordinates)
        else:
            print(
                "You need to pass the arguments track or location_coordinates for the function to work"
            )
            return None
    except requests.exceptions.ConnectionError:
        print("Error on requests package, starting again...")
        time.sleep(600)
        pass
    except KeyboardInterrupt:
        print("Why would you do that???????")


def get_all_tweets_from_user(screen_name, secret_file_path):
    secret_json = secret_file_path
    secrets = read_secret_file(secret_json)
    consumer_key = secrets["consumer_key"]
    consumer_secret = secrets["consumer_secret"]
    access_token = secrets["access_key"]
    access_token_secret = secrets["access_token_secret"]

    # Twitter only allows access to a users most recent 3240 tweets with this method

    # authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    # initialize a list to hold all the tweepy Tweets
    alltweets = []

    # make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name=screen_name, count=200)

    # save most recent tweets
    alltweets.extend(new_tweets)

    # save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    # keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        print("getting tweets before %s" % (oldest))

        # all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(
            screen_name=screen_name, count=200, max_id=oldest
        )

        # save most recent tweets
        alltweets.extend(new_tweets)

        # update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        print("...%s tweets downloaded so far" % (len(alltweets)))
        time.sleep(30)

    # transform the tweepy tweets into a 2D array that will populate the csv
    adict = dict()
    adict["id"] = []
    adict["created_at"] = []
    adict["text"] = []
    for tweet in alltweets:
        adict["id"].append(tweet.id_str)
        adict["created_at"].append(tweet.created_at)
        adict["text"].append(tweet.text.encode("utf-8"))

    return pd.DataFrame(data=adict, columns=adict.keys())


def store_data(
    file_to_save,
    _screen_name,
    _full_text,
    _coordinates,
    _created_at,
    _favorite_count,
    _id,
    _lang,
    _retweet_count,
    _source,
    _media_type,
    _media_url,
    _user_pic,
):
    """
    Populates the database with status updates from Twitter search.

    :param _media_url: URL containing media displayed in tweet
    :param _media_type: Type of media (currently supports photo only)
    :param _source: What device was used to post the tweet
    :param _retweet_count: Number of times the status was retweeted
    :param _lang: Language
    :param _id: User numerical id code
    :param _created_at: Time when was created
    :param _coordinates: Coordinates of tweet location
    :param _full_text: Full text of status update
    :param _screen_name: User's display name
    :param _favorite_count: Number of times the tweet has been liked
    """
    f = open(file_to_save, "w+")
    f.write(
        ",".join(
            [
                _screen_name,
                _full_text,
                _coordinates,
                _created_at,
                _favorite_count,
                _id,
                _lang,
                _retweet_count,
                _source,
                _media_type,
                _media_url,
                _user_pic,
            ]
        )
    )
    f.close()


def check_for_none(item):
    """

    :param item:
    :return:
    """
    if item is None:
        return "null"
    else:
        try:
            item = unicodedata.normalize("NFKD", item).encode("utf-8", "ignore")
            return item
        except:
            return item


def rest_api_tweets_download_csv(
    secret_file_path, subject, tweet_count, geocode=[], time_interval=[]
):
    """
    Retrieves status updates from Twitter and registers them in a database.
    :param tweet_count: Number of tweets PER PAGE. If omitted, default = 15. Max = 200.
    :param subject: subject of tweet search
    """

    secret_json = secret_file_path
    secrets = read_secret_file(secret_json)
    consumer_key = secrets["consumer_key"]
    consumer_secret = secrets["consumer_secret"]
    access_token = secrets["access_key"]
    access_token_secret = secrets["access_token_secret"]

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(
        auth,
        wait_on_rate_limit_notify=True,
        wait_on_rate_limit=True,
        retry_count=3,
        retry_delay=180,
        retry_errors=set([401, 404, 500, 503]),
    )  # auto download limit control

    counter = 0
    if len(geocode) > 0:
        arg_geo = ",".join(geocode)
    else:
        arg_geo = None

    if len(time_interval) > 0:
        arg_time = time_interval
    else:
        arg_time = [None, None]

    try:
        alist = tweepy.Cursor(
            api.search,
            q=subject,  # this is the search subject
            count=tweet_count,
            result_type="recent",
            include_entities=True,
            monitor_rate_limit=True,
            lang="en",
            geocode=arg_geo,  # "lat,long,radius" (radius must be “mi” or “km”)
            since=arg_time[
                0
            ],  # YYYY-DD-MM date when collection begins(max 7 days back in time)
            until=arg_time[1],  # YYYY-DD-MM date when collection stops
            tweet_mode="extended",
        ).items()
    except:
        print("Error")
        alist = []

    for status in alist:  # change to extended mode to include media in entities

        try:
            # Grab wanted data from tweet
            status_user = str(check_for_none(status.user.screen_name.encode("utf-8")))
            status_user_pic = str(check_for_none(status.user.profile_image_url_https))
            status_text = str(check_for_none(status.full_text).encode("utf-8"))
            status_coordinates = check_for_none(status.coordinates)
            status_creation_time = datetime.strftime(
                status.created_at, "%Y-%m-%d %H:%M:%S"
            )
            status_favorite_count = status.favorite_count
            status_user_id = int(status.id)
            status_lang = str(status.lang.encode("utf-8"))
            status_retweet_count = int(status.retweet_count)
            status_source = str(check_for_none(status.source.encode("utf-8")))
            if "media" in status.entities.keys():
                status_media_type = str(
                    check_for_none(status.entities["media"][0]["type"].encode("utf-8"))
                )
                status_media_url = str(
                    check_for_none(
                        status.entities["media"][0]["media_url"].encode("utf-8")
                    )
                )
            else:
                status_media_type = "null"
                status_media_url = "null"

            store_data(
                subject,
                status_user,
                status_text,
                status_coordinates,
                status_creation_time,
                status_favorite_count,
                status_user_id,
                status_lang,
                status_retweet_count,
                status_source,
                status_media_type,
                status_media_url,
                status_user_pic,
            )
            counter += 1
        except tweepy.TweepError as e:
            print(e)
            print("Error found. Retrying in 60 seconds.")
            print(
                "Localtime: "
                + time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime())
            )
            time.sleep(60)
            continue
            # tweet_hunt(search, 200)

        except MemoryError:
            print(
                "Memory Error found on "
                + time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime())
            )
            print("Retrying in 5 minutes.")
            time.sleep(300)
            continue
            # tweet_hunt(search, 200)

        except Exception as e2:
            print(
                "Error on line {}".format(sys.exc_info()[-1].tb_lineno),
                type(e2).__name__,
                e2,
            )
            # print "Exception found. Retrying in 60 seconds."
            # print "Localtime: " + time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime())
            # time.sleep(60)
            # tweet_hunt(search, 200)
            continue

        except KeyboardInterrupt:
            print("Thank you for hunting tweets with us!")
            break

        if counter % 10000 == 0:
            print("Number of entries in database: ", counter)
