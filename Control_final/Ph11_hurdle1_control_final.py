#Kapil Sinha
#10/24/16
#Ph 11 Hurdle 1

import random
import numpy
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

INITIAL_NUM_BIKES = 13 #number of bikes per station initially
INITIAL_PRICE = 1.50
NUM_TERMINALS = 32 #constant number of terminals in each station

def generateCity():
    '''Creates the city: 1000x1000 Point array with random Points assigned popularity values, 34x34 Stations with 105 extra randomly placed Stations, and 245,000 randomly placed people'''
    city_map = [[Point(x,y) for x in range(1000)] for y in range(1000)]
    #2-D array of the city map where each Point is a 10m x 10m square
    assignPopularity(city_map) #set popularity values for random Points
    for i in range(5,996,30): #place 34x34 stations spaced by 30 (x 10 m)
        for j in range(5,996,30):
            city_map[i][j].station = Station(i,j)
            city_map[i][j].incrementNumStations()
    for k in range(105): #randomly place remaining 105 stations
        x = random.randint(0,999)
        y = random.randint(0,999)
        city_map[x][y].station = Station(x,y)
        city_map[x][y].incrementNumStations()
    for i in range(245000): #randomly place 245,000 people (Bike-Share users)
        x = random.randint(0,999)
        y = random.randint(0,999)
        city_map[x][y].userlist.append(User(x,y))
        city_map[x][y].incrementNumPeople()
    return city_map

def assignPopularity(city_map):
    '''Assigns popularity values to randomly chosen locations such that there are 80,000 Points with popularity 1, 4,000 3 x 3 Point squares with popularity 10, 200 5 x 5 Point squares with popularity 100, and 10  7 x 7 Point squares with popularity 1,000, and 1 9 x 9 Point square with popularity 5,000'''
    #Note: we are adding these popularity values to Points with replacement; that is, a Point that is assigned a certain popularity may be overrided by the next statement; however, since we are assigning increasing popularity values and some randomness is acceptable, this should not cause any errors
    for a in range(80000): #(Approximately) 80,000 10m x 10m squares with popularity 1
        x = random.randint(0,999)
        y = random.randint(0,999)
        city_map[x][y].popularity = 1
    for b in range(4000): #4,000 30m x 30m squares with popularity 10
        x = random.randint(1,998)
        y = random.randint(1,998)
        for k in range(x-1,x+2):
            for l in range(y-1,y+2):
                city_map[k][l].popularity = 10
    for c in range(200): #200 50m x 50m squares with popularity 100
        x = random.randint(2,997)
        y = random.randint(2,997)
        for m in range(x-2,x+3):
            for n in range(y-2,y+3):
                city_map[m][n].popularity = 100
    for d in range(10): #10 70m x 70m squares with popularity 1000
        x = random.randint(3,996)
        y = random.randint(3,996)
        for o in range(x-3,x+4):
            for p in range(y-3,y+4):
                city_map[o][p].popularity = 1000
    for e in range(1): #1 90m x 90m square with popularity 5,000 i.e. Eiffel Tower
        x = random.randint(4,997)
        y = random.randint(1,998)
        for q in range(x-4,x+5):
            for r in range(y-4,y+5):
                city_map[q][r].popularity = 5000

def numTripsinHour(t):
    """Description: Returns the total number of trips in an hour given the time
    Input Argument: time t of day (0-24)
    Return Value: int"""
    if t >= 0 and t < 1:
        return 3783
    if t >= 1 and t < 2:
        return 2648
    if t >= 2 and t < 3:
        return 1892
    if t >= 3 and t < 4:
        return 1135
    if t >= 4 and t < 5:
        return 946
    if t >= 5 and t < 6:
        return 757
    if t >= 6 and t < 7:
        return 946
    if t >= 7 and t < 8:
        return 3026
    if t >= 8 and t < 9:
        return 7566
    if t >= 9 and t < 10:
        return 6053
    if t >= 10 and t < 11:
        return 4540
    if t >= 11 and t < 12:
        return 5296
    if t >= 12 and t < 13:
        return 6053
    if t >= 13 and t < 14:
        return 6053
    if t >= 14 and t < 15:
        return 6053
    if t >= 15 and t < 16:
        return 6053
    if t >= 16 and t < 17:
        return 6809
    if t >= 17 and t < 18:
        return 8701
    if t >= 18 and t < 19:
        return 10592
    if t >= 19 and t < 20:
        return 9836
    if t >= 20 and t < 21:
        return 7188
    if t >= 21 and t < 22:
        return 4918
    if t >= 22 and t < 23:
        return 4540
    if t >= 23 and t < 24:
        return 4540
    if t>=24 and t < 25:
        return 50 # This is done purely for testing purposes (running this program over t=0 to t=24 is computationally heavy)
    else:
        raise ValueError('Time must be between 0 and 24')

def incrementTime(t):
    '''Increases t by 1 hour unless it is between 23 and 24, in which case it returns a number between 0 and 1 (goes to the next day)'''
    if t >= 0 and t <23:
        return t+1
    elif t >= 23:
        return (t+1)-24
    else:
        raise ValueError('Time cannot be negative')

def findBikingUsers(city_map, t):
    '''Returns a random subset of Users of a size depending on t. We can randomly choose n people because our time step is 1 hour and so we don't need to consider distance from a station as a factor of WHETHER a user will rent a bike (as was done in literature with a kinked curve); moreover, we evenly spaced out most of the stations, so distance from stations is relatively constant. Instead, the distance from a station will be a factor of WHICH station the biker will rent from'''
    user_list = [] #list of all users
    for x in range(1000):
        for y in range(1000):
            user_list += city_map[x][y].userlist 
    return random.sample(user_list, numTripsinHour(t))
    # return a random sample of size numTripsinHour(t) of all the Users

def calcManhattanDistance(x_origin,y_origin,x_destination,y_destination): 
    '''Manhattan distanc is a factor of the likelihood that a user picks up a bike'''
    return abs(x_origin - x_destination) + abs(y_origin - y_destination)

def calcProbabilityBikeAvailability(bike_availability, isRentStation):
    '''Calculates the probability that a certain user will use a station depending on the average bike availability, using a basic kinked function. Will be used in calcRentStation() and calcDropOffStation()
bike_availabliity - integer returned from averageBikeAvailability()
isRentStation - True if used in calcRentStation, False if used in calcDropOffStation'''
    if isRentStation == True:
        if bike_availability >= 5:
            return 1.0 #100% probability i.e. bike availability is not a factor in the user's decision-making if there are more than 5 bikes in a station
        else:
            return 0.2*bike_availability # if there are fewer than 5 bikes in that area, the probability that a user will go to that station decreases linearly (such that if there are 0 bikes on average, there is no chance that the user will go there to rent a bike
    else: #if DropOffStation
        if bike_availability <= 27: #5 less than the number of terminals
            return 1.0
        else: #if the average station in the area is almost full, the chance a user will drop off a bike there decreases
            return 0.2* bike_availability

def calcProbabilityManhattanDistance(manhattan_distance, isWalkingDistance): 
    '''Calculates the probability that a certain user will use a station depending on the walking Manhattan distance to the station (for renting a bike) and biking Manhattan distance to a station (for dropping off a a bike)'''
    if isWalkingDistance == True:
        if manhattan_distance < 300:
            return 1 - (.00252 * manhattan_distance)
        else:
            return max(0,1 - (.00252 * manhattan_distance + .01367 * (manhattan_distance - 300))) #zero probability of walking to a station over 315 m away
    else: #isBikingDistance #WRITE LOGIC FOR THESE NUMBERS IN THE PAPER
        if manhattan_distance < 3000:
            return 1 - (.000252 * manhattan_distance)
        else:
            return max(0,1 - (.000252 * manhattan_distance + .001367 * (manhattan_distance - 3000))) #zero probability of biking to a station over 3150 m away

def calcProbabilityPopularity(city_map,station): 
    '''Calculates the probability that a certain user will use a station depending on the surrounding points of interest. Uses the Manhattan distance calculation from calcProbailityManhattanDistance() function to find the popularity in a certain distance from the station (because the biker then has to walk to the point of interest)'''
    popularity_in_area = 0
    #we could cycle through the entire city_map but we know from our previous Manhattan distance assumptions that no user will walk more than 315 m (rounding up to 320 m) to arrive at his/her destination so this makes our computation faster
    for i in range(max(0,station.x-32),min(station.x+33,999)): 
        for j in range(max(0,station.y-32),min(station.y+33,999)): 
            popularity_in_area += (calcProbabilityManhattanDistance(calcManhattanDistance(station.x,station.y,i,j), True)*city_map[i][j].popularity)
    return popularity_in_area

def calcProbabilityPrice(price): #price of picking up/dropping off at a station is in dollars
    '''Returns the probability that a user will use a certain station based on price. We extend the supply/demand curve for monopolistic competition (a straight line) such that probability (replacing quantity demanded) is 1 at 0 dollars and 0 at 3 dollars per station (note that we are charging at both the rental station and the dropoff station).'''
    if (1 - price/3) < 0:
        return 0
    return 1 - (price/3)

def calcRentStation(city_map, user, station_list):
    '''Determines which station a biking user will rent from. Uses Manhattan distance, average bike availability, and price'''
    station_probability = [] #We are creating a list of probabilities whose index matches those of the stations; a dictionary may be more appropriate here but the following is simple for a list 
    for i in range(len(station_list)):
        bike_availability_probability = calcProbabilityBikeAvailability(station_list[i].averageBikeAvailability(city_map),True)
        manhattan_distance_probability = calcProbabilityManhattanDistance(calcManhattanDistance(user.x,user.y,station_list[i].x,station_list[i].y), True)
        price_probability = calcProbabilityPrice(station_list[i].rental_price)
        station_probability.append(bike_availability_probability * manhattan_distance_probability * price_probability)
    station_probability_normalized = normalizeList(station_probability)
    rent_station = numpy.random.choice(station_list, 1, p=station_probability_normalized)[0] #randomly choose a station based on the calculated probabilities (allows for randomness that would exist in the real world with the necessary constraints)
    return rent_station

def calcDropOffStation(city_map, user, station_list):
    '''Determines at which station a biking user will drop off a bike. Uses popularity, Manhattan distance, average bike availability, and price'''
    station_probability = [] #We are creating a list of probabilities whose index matches those of the stations (like in calcRentStation)
    for i in range(len(station_list)):
        bike_availability_probability = calcProbabilityBikeAvailability(station_list[i].averageBikeAvailability(city_map),False)
        popularity_probability = calcProbabilityPopularity(city_map,station_list[i])
        manhattan_distance_probability = calcProbabilityManhattanDistance(calcManhattanDistance(user.x,user.y,station_list[i].x,station_list[i].y), False)
        price_probability = calcProbabilityPrice(station_list[i].dropoff_price)
        station_probability.append(bike_availability_probability * manhattan_distance_probability * price_probability * popularity_probability)
    station_probability_normalized = normalizeList(station_probability)
    dropoff_station = numpy.random.choice(station_list, 1, p=station_probability_normalized)[0] #randomly choose a station based on the calculated probabilities
    return dropoff_station
    

def normalizeList(lst):
    '''Helper function used in calcRentStation() and calcDropOffStation(). Returns a list whose elements add up to 1. Divides every element in a list by the sum of the list'''
    lst_sum = 0
    for i in range(len(lst)):
        lst_sum += lst[i]
    lst_normalized = []
    for j in range(len(lst)):
        lst_normalized.append(float(lst[j])/float(lst_sum))
    lst_normalized_sum = 0
    for k in range(len(lst_normalized)):
        lst_normalized_sum += lst_normalized[k]
    return lst_normalized

def getStationList(city_map):
    '''Returns a list of all the stations. Used in calcRentStation() and calcDropOffStation().'''
    station_list = [] #list of all stations
    for x in range(1000):
        for y in range(1000):
            if city_map[x][y].num_stations > 0:
                station_list.append(city_map[x][y].station)
    return station_list

def adjustStationPrice(station, isRentBike):
    '''Adjusts price of dropping off and renting a bike from a station based on the number of bikes. The actual numbers that have been created as constants i.e. INITIAL_NUM_BIKES, INITIAL_PRICE, and NUM_TERMINALS have been used here so that the functions are easier to calculate and understand'''
    if isRentBike == True:
        if station.num_bikes < 13:
            station.price = (-1.5/13)*(station.num_bikes)+3
        elif station.num_bikes >= 13 and station.num_bikes < 18:
            station.price = (-1.5/19)*(station.num_bikes)+((32*1.5)/19)
        else: #num_bikes >= 18
            station.price = -(1.0/28)*(station.num_bikes-18)+0.5
    else: #isRentBike == False
        if station.num_bikes < 13:
            station.price = (1.5/13)*(station.num_bikes)
        elif station.num_bikes >= 13 and station.num_bikes < 20:
            station.price = (1.5/19)*(station.num_bikes)+(3-((1.5*32)/19))
        else: #num_bikes >= 20
            station.price = (1.0/280)*(station.num_bikes-18)+2.95
   
def visualizeCity(station_list,t):
    '''Plots a 3-D graph of the x and y values of the station along with the number of bikes on the z axis'''
    X = [] #x_list
    Y = [] #y_list
    Z = [] #num_bikes_list
    for station in station_list:
        X.append(station.x)
        Y.append(station.y)
        Z.append(station.num_bikes)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = numpy.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False) #, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0) #, antialiased=False)
    ax.set_zlim(0, 32)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    image_name = 'Paristopia_final_Control_t='+str(t)+'.png'
    plt.savefig(image_name, bbox_inches='tight')
    #plt.show()

def visualizeCity2(station_list):
    '''Returns a condensed 2-D array of city_map with just the stations for easy visualization of changes in number of bikes in each station''' #OUTDATED - REPLACED BY visualizeCity()
    condensed_city = []
    for station in station_list:
        condensed_city.append((station.x,station.y,station.num_bikes,station.rental_price,station.dropoff_price))
    print condensed_city

def numBikesDifference(station_list):
    '''Returns the average difference between a station's initial bike count (13) and its current number of bikes for all the stations). The higher this difference, the more uneven the number of bikes in station (and the worse the source/sink problem)'''
    num_bikes_difference = 0
    for i in range(len(station_list)):
        num_bikes_difference += abs(station_list[i].num_bikes - INITIAL_NUM_BIKES)
    return float(num_bikes_difference)/len(station_list)

class Point:
    '''Each Point object (there are 1,000,000) represents a 10 m x 10 m area and contains: 
    x - x coordinate of location
    y - y coordinate of location
    popularity - popularity of location
    num_stations - number of stations (greater than 1 only if a randomly assigned station is in an area where there already is another station)
    num_people - number of people (Bike-Share users) in area'''
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.popularity = 0 #every Point is initialized with 0 "popularity" but some will be given certain values if they are points of interest
        self.station = None #variable station may hold Station object but is not initialized with one
        self.num_stations = 0 #a point is initialized with 0 stations
        self.userlist = [] #list userlist may contain several User objects but is initialized empty
        self.num_people = 0 #a point is initialized with 0 people (Bike-Share users)
    def incrementNumStations(self):
        self.num_stations += 1
    def incrementNumPeople(self):
        self.num_people += 1


class Station:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.num_bikes = INITIAL_NUM_BIKES
        self.num_terminals = NUM_TERMINALS
        self.rental_price = INITIAL_PRICE
        self.dropoff_price = INITIAL_PRICE
    def incrementNumBikes(self):
        self.num_bikes+=1
    def decrementNumBikes(self):
        self.num_bikes-=1
    def averageBikeAvailability(self,city_map): 
        num_bikes_in_area = 0
        num_stations_in_area = 0
        for x in range(self.x-30,self.x+30):
            if x < 0: #make sure that x is between 0 and 999
                x = 0
            if x >= 1000:
                x = 999
            for y in range(self.y-30,self.y+30):
                if y < 0: #make sure that y is between 0 and 999
                    y = 0
                if y >= 1000:
                    y = 999
                if city_map[x][y].num_stations != 0:
                    num_stations_in_area += 1
                    num_bikes_in_area += city_map[x][y].station.num_bikes
        return float(num_bikes_in_area)/float(num_stations_in_area)
    #returns the average number of bikes in a station's area
    #(600 m by 600 m square with that station as the center)
    #at a given time


class User:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def rentBike(self, rent_station): #action taken when user decides to rent a bike
        #note that the user is not moved when he/she rents a bike; this simplification allows us to ignore random movements of bike users so we assume that bike renters are uniformly distributed
        rent_station.decrementNumBikes()
    def dropOffBike(self, dropoff_station): #action taken when user decides to drop off a bike
        #note that the user is not moved when he/she drops off a bike; we assume that bikers are uniformly distributed 
        dropoff_station.incrementNumBikes()

'''#Experiment - changing prices
city_map = generateCity()
station_list = getStationList(city_map)
visualizeCity(station_list,-1)
for t in range(24):
    print t
    biking_users = findBikingUsers(city_map, 24) #find a way to increment the time and measure results
    for user in biking_users:
        station = calcRentStation(city_map,user,station_list)
        user.rentBike(station)
        adjustStationPrice(station,True)
    for user in biking_users:
        station = calcDropOffStation(city_map,user,station_list)
        user.dropOffBike(station)
        adjustStationPrice(station,False)
    visualizeCity(station_list,t)
    print numBikesDifference(station_list)
'''

#Control:
city_map = generateCity()
station_list = getStationList(city_map)
visualizeCity(station_list,-1)
for t in range(24):
    print t
    biking_users = findBikingUsers(city_map, t) #change t to 24 to run a simulation with 100 users
    for user in biking_users:
        user.rentBike(calcRentStation(city_map,user,station_list))
    for user in biking_users:
        user.dropOffBike(calcDropOffStation(city_map,user,station_list))
    visualizeCity(station_list,t)
    print numBikesDifference(station_list)
 
