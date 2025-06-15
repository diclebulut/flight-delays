import pandas as pd

airlines = pd.read_csv('Inputs/airlines.csv')
airports = pd.read_csv('Inputs/airports.csv')
flights = pd.read_csv('Inputs/flights.csv')


flights = flights.merge(airports[['IATA_CODE', 'CITY','STATE', 'COUNTRY', 'LATITUDE', 'LONGITUDE']], left_on = 'ORIGIN_AIRPORT', right_on = 'IATA_CODE', how='left', suffixes=('', '_dep') )
flights = flights.merge(airports[['IATA_CODE', 'CITY','STATE', 'COUNTRY', 'LATITUDE', 'LONGITUDE']], left_on = 'DESTINATION_AIRPORT', right_on = 'IATA_CODE', how='left', suffixes=('', '_arr') )

def hhmm_to_time(time_float):
    if pd.isna(time_float):
        return None
    
    # Convert to integer and pad with zeros
    time_str = f"{int(time_float):04d}"
    hours = time_str[:2]
    minutes = time_str[2:]
    return f"{hours}:{minutes}"

flights['DEPARTURE_TIME'] = flights['DEPARTURE_TIME'].apply(hhmm_to_time)
flights['SCHEDULED_DEPARTURE'] = flights['SCHEDULED_DEPARTURE'].apply(hhmm_to_time)
flights['SCHEDULED_ARRIVAL'] = flights['SCHEDULED_ARRIVAL'].apply(hhmm_to_time)
flights['ARRIVAL_TIME'] = flights['ARRIVAL_TIME'].apply(hhmm_to_time)
flights['WHEELS_OFF'] = flights['WHEELS_OFF'].apply(hhmm_to_time)
flights['WHEELS_ON'] = flights['WHEELS_ON'].apply(hhmm_to_time)

