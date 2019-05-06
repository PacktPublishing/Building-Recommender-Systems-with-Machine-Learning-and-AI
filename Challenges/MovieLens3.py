import os
import csv
import sys
import re
import pandas as pd
import numpy as np

from surprise import Dataset
from surprise import Reader

from collections import defaultdict

class MovieLens:

    movieID_to_name = {}
    name_to_movieID = {}
    ratingsPath = '../ml-latest-small/ratings.csv'
    moviesPath = '../ml-latest-small/movies.csv'
    

    
    def loadMovieLensLatestSmall(self, outlierStdDev = 3.0):

        # Look for files relative to the directory we are running from
        os.chdir(os.path.dirname(sys.argv[0]))

        ratingsDataset = 0
        
        ratings = pd.read_csv(self.ratingsPath, encoding='latin-1')
        print("Raw ratings data:")
        print(ratings.head())
        print(ratings.shape)
        
        ratingsByUser = ratings.groupby('userId', as_index=False).agg({"rating": "count"})
        print("Ratings by user:")
        print (ratingsByUser.head())

        ratingsByUser['outlier'] = (abs(ratingsByUser.rating - ratingsByUser.rating.mean()) > ratingsByUser.rating.std() * outlierStdDev)
        ratingsByUser = ratingsByUser.drop(columns=['rating'])
        print("Users with outliers computed:")
        print (ratingsByUser.head())

        combined = ratings.merge(ratingsByUser, on='userId', how='left')
        print("Merged dataframes:")
        print(combined.head())
        
        filtered = combined.loc[combined['outlier'] == False]
        filtered = filtered.drop(columns=['outlier', 'timestamp'])
        print("Filtered ratings data:")
        print (filtered.head())
        print (filtered.shape)
        
        reader = Reader(rating_scale=(1, 5))
        ratingsDataset = Dataset.load_from_df(filtered, reader)

        self.movieID_to_name = {}
        self.name_to_movieID = {}

        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
                movieReader = csv.reader(csvfile)
                next(movieReader)  #Skip header line
                for row in movieReader:
                    movieID = int(row[0])
                    movieName = row[1]
                    self.movieID_to_name[movieID] = movieName
                    self.name_to_movieID[movieName] = movieID

        return ratingsDataset
    
    def getNewMovies(self):
        newMovies = []
        years = self.getYears()
        # What's the newest year in our data?
        latestYear = max(years.values())
        print ("Newest year is ", latestYear)
        for movieID, year in years.items():
            if year == latestYear:
                newMovies.append(movieID)
                #print (self.getMovieName(movieID))
        return newMovies

    def getUserRatings(self, user):
        userRatings = []
        hitUser = False
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                userID = int(row[0])
                if (user == userID):
                    movieID = int(row[1])
                    rating = float(row[2])
                    userRatings.append((movieID, rating))
                    hitUser = True
                if (hitUser and (user != userID)):
                    break

        return userRatings

    def getPopularityRanks(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                movieID = int(row[1])
                ratings[movieID] += 1
        rank = 1
        for movieID, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[movieID] = rank
            rank += 1
        return rankings
    
    def getGenres(self):
        genres = defaultdict(list)
        genreIDs = {}
        maxGenreID = 0
        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
            movieReader = csv.reader(csvfile)
            next(movieReader)  #Skip header line
            for row in movieReader:
                movieID = int(row[0])
                genreList = row[2].split('|')
                genreIDList = []
                for genre in genreList:
                    if genre in genreIDs:
                        genreID = genreIDs[genre]
                    else:
                        genreID = maxGenreID
                        genreIDs[genre] = genreID
                        maxGenreID += 1
                    genreIDList.append(genreID)
                genres[movieID] = genreIDList
        # Convert integer-encoded genre lists to bitfields that we can treat as vectors
        for (movieID, genreIDList) in genres.items():
            bitfield = [0] * maxGenreID
            for genreID in genreIDList:
                bitfield[genreID] = 1
            genres[movieID] = bitfield            
        
        return genres
    
    def getYears(self):
        p = re.compile(r"(?:\((\d{4})\))?\s*$")
        years = defaultdict(int)
        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
            movieReader = csv.reader(csvfile)
            next(movieReader)
            for row in movieReader:
                movieID = int(row[0])
                title = row[1]
                m = p.search(title)
                year = m.group(1)
                if year:
                    years[movieID] = int(year)
        return years
    
    def getMiseEnScene(self):
        mes = defaultdict(list)
        with open("LLVisualFeatures13K_Log.csv", newline='') as csvfile:
            mesReader = csv.reader(csvfile)
            next(mesReader)
            for row in mesReader:
                movieID = int(row[0])
                avgShotLength = float(row[1])
                meanColorVariance = float(row[2])
                stddevColorVariance = float(row[3])
                meanMotion = float(row[4])
                stddevMotion = float(row[5])
                meanLightingKey = float(row[6])
                numShots = float(row[7])
                mes[movieID] = [avgShotLength, meanColorVariance, stddevColorVariance,
                   meanMotion, stddevMotion, meanLightingKey, numShots]
        return mes
    
    def getMovieName(self, movieID):
        if movieID in self.movieID_to_name:
            return self.movieID_to_name[movieID]
        else:
            return ""
        
    def getMovieID(self, movieName):
        if movieName in self.name_to_movieID:
            return self.name_to_movieID[movieName]
        else:
            return 0