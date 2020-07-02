from django.shortcuts import render
import tweepy
from textblob import TextBlob  
import simplejson as json
from matplotlib import pyplot as plt 
from langdetect import detect

# Create your views here.
def home(request):
    return render(request,'home.html')

def Charts(request):
    return render(request,'result1.html')   

def getLang(langList):
    dict = {}
    

def getAnalysis(Analysis):
    name = ['positive','nutral','nagative']
    count = [0,0,0]
    for i in Analysis:
        if i<0:
            count[2]+=1
        elif i==0:
            count[1]+=1
        else :
            count[0]+=1
    plt.bar(name,count)
    #plt.bar(name, count, align='center', alpha=0.5)
    plt.ylabel('Count')
    plt.title('Sentiment')
    plt.savefig(r"static\Analysis.png") 
    plt.clf()
    return   

def getLang(Analysis):
    name2 = ['English','Hindi','French','Japanesh','Chinesh','Russian','Other']
    count2 = [0,0,0,0,0,0,0]
    for i in Analysis:
        if i=='en':
            count2[0]+=1
        elif i=='hi':
            count2[1]+=1
        elif i=='fr':
            count2[2]+=1
        elif i=='ja':
            count2[3]+=1
        elif i=='zh-cn':
            count2[4]+=1
        elif i=='ru':
            count2[5]+=1                
        else :
            count2[6]+=1
            
    plt.bar(name2,count2)
    #plt.bar(name, count, align='center', alpha=0.5)
    plt.ylabel('Count')
    plt.title('Languages Analysis')
    plt.savefig(r"static\LangAnaly.png") 
    plt.clf()
    return


def getVarified(Analysis):
    name2 = ['Verified', 'Unverified','Data Unavailable']
    count2 = [0, 0, 0]
    for i in Analysis:
        if i == 'true':
            count2[0] += 1
        elif i == 'false':
            count2[1] += 1
        else :
            count2[2] +=1


    plt.bar(name2, count2)
    # plt.bar(name, count, align='center', alpha=0.5)
    plt.ylabel('Count')
    plt.title('Account Analysis')
    plt.savefig(r"static\AccountAnaly.png")
    plt.clf()
    return

def getCouontFollower(Analysis):
    name2 = ['1-100', '101-200', '201-300', '301-400', '401-500', '501-600', '600+']
    count2 = [0, 0, 0, 0, 0, 0, 0]
    for i in Analysis:
        if i >= 0 and i <= 100:
            count2[0] += 1
        elif i >= 101 and i <= 200:
            count2[1] += 1
        elif i >= 201 and i <= 300:
            count2[2] += 1
        elif i >= 301 and i <= 400:
            count2[3] += 1
        elif i >= 401 and i <= 500:
            count2[4] += 1
        elif i >= 501 and i <= 600:
            count2[5] += 1
        else:
            count2[6] += 1
    plt.bar(name2, count2)
    # plt.bar(name, count, align='center', alpha=0.5)
    plt.ylabel('Count')
    plt.title('No of Followers User have')
    plt.savefig(r"static\FollowersAnaly.png")
    plt.clf()
    return


def getNoFollowing(Analysis):
    name2 = ['1-100','101-200','201-300','301-400','401-500','501-600','600+']
    count2 = [0, 0,0,0,0,0,0]
    for i in Analysis:
        if i >=0 and i<=100:
            count2[0] += 1
        elif i >=101 and i<=200:
            count2[1] += 1
        elif i >=201 and i<=300:
            count2[2] += 1
        elif i >=301 and i<=400:
            count2[3] += 1
        elif i >=401 and i<=500:
            count2[4] += 1
        elif i >=501 and i<=600:
            count2[5] += 1
        else:
            count2[6] += 1
    plt.bar(name2, count2)
    # plt.bar(name, count, align='center', alpha=0.5)
    plt.ylabel('Count')
    plt.title('No of Following User have')
    plt.savefig(r"static\FollowingAnaly.png")
    plt.clf()
    return

def getNoTweets(Analysis):
    name2 = ['1-250', '215-500', '501-750', '751-1000', '1001-1250', '1251-1500', '1500+']
    count2 = [0, 0, 0, 0, 0, 0, 0]
    for i in Analysis:
        if i >= 0 and i <= 250:
            count2[0] += 1
        elif i >= 251 and i <= 500:
            count2[1] += 1
        elif i >= 501 and i <= 750:
            count2[2] += 1
        elif i >= 751 and i <= 1000:
            count2[3] += 1
        elif i >= 1001 and i <= 1250:
            count2[4] += 1
        elif i >= 1251 and i <= 1500:
            count2[5] += 1
        else:
            count2[6] += 1
    plt.figure(figsize=(10,5))
    plt.bar(name2, count2)
    #plt.bar(name, count, align='center', alpha=0.5)
    plt.ylabel('Count')
    plt.title('No of tweets by particular user')
    plt.savefig(r"static\NoOfTweetsAnaly.png")
    plt.clf()
    return











def result(request):
    no_of_tweets = 20
    s=request.GET['search']
    print(s)
    consumer_key='81vF03XpNUTi3TlKkskujt35D'
    consumer_secret='haIjHArQTxndszj5Q9Vfs6HoASSLTCBwqDDbOsElfxd7Xw6YIl'
    access_token='1218525746697658368-ph7yuWluqdsMPtIWzrlSxsA1Z8SN5q'
    access_token_secret='smQXZ5iMk7vHzzCpyWjK708rHQN6BoqBvIv9uhpg0kP9w'
    auth=tweepy.OAuthHandler(consumer_key,consumer_secret)
    auth.set_access_token(access_token,access_token_secret)
    api=tweepy.API(auth)
    public_tweets=tweepy.Cursor(api.search,q=s).items(no_of_tweets)
    lsLocation = []
    lsTweets = []
    lsLang = []
    lsAge = []
    lsAnalysis = []
    lsVerified = []
    lsFollower = []
    lsFollowing = []
    lsTweetsCount = []
    lsConti = set()
    setLang = set()
    setLoca = set()
    for tweet in public_tweets:
        #print(tweet.text)
        analysis=TextBlob(tweet.text)
        #print(analysis)
        lsAnalysis.append(analysis.sentiment.polarity)
        lsLocation.append(tweet.user.location)
        setLoca.add(tweet.user.location)
        lsTweets.append(tweet.text)
        lsVerified.append(tweet.user.verified)
        lsFollower.append(tweet.user.followers_count)
        lsFollowing.append(tweet.user.friends_count)
        lsTweetsCount.append(tweet.user.statuses_count)
        lsConti.add(tweet.user.id)
        #lsAge.append(tweet.user.age)
        tg = detect(tweet.text)
        lsLang.append(tg)
        setLang.add(tg)
        #print(tweet.user)
        #sdata = json.dumps(tweet.__json) 
        #print("###################################",tweet.user.location,"#####################################")
    getAnalysis(lsAnalysis)
    getLang(lsLang)
    getVarified(lsVerified)
    getNoFollowing(lsFollowing)
    getCouontFollower(lsFollower)
    getNoTweets(lsTweetsCount)
    #print(lsLang)
    Info = {}
    Info['location'] = lsLocation
    Info['no_of_tweets'] = no_of_tweets
    #Info['tweets'] = lsTweets
    Info['languages'] = lsLang 
    Info['Analysis'] = lsAnalysis
    Info['Analysis_Img'] = 'Analysis.png'
    Info['Contribut'] = len(lsConti) 
    Info['TotLang'] = len(setLang)
    Info['TotLoca'] = len(setLoca) 
    
    return render(request,'result.html',{'tweets':lsTweets,'keyword':s,'result':Info})