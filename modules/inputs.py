import pandas as pd

airlines = pd.read_csv('Inputs/airlines.csv')
airports = pd.read_csv('Inputs/airports.csv')
flights = pd.read_csv('Inputs/flights.csv')
tail_numbers = pd.read_csv('Inputs/Aircraft_Tail_Numbers_and_Models_at_SFO.csv')


flights = flights.merge(airports[['IATA_CODE', 'CITY','STATE', 'COUNTRY', 'LATITUDE', 'LONGITUDE']], left_on = 'ORIGIN_AIRPORT', right_on = 'IATA_CODE', how='left')
flights = flights.rename(columns={
    'IATA_CODE': 'IATA_CODE_dep',
    'CITY': 'CITY_dep',
    'STATE': 'STATE_dep',
    'COUNTRY': 'COUNTRY_dep',
    'LATITUDE': 'LATITUDE_dep',
    'LONGITUDE': 'LONGITUDE_dep'
})
flights = flights.merge(airports[['IATA_CODE', 'CITY','STATE', 'COUNTRY', 'LATITUDE', 'LONGITUDE']], left_on = 'DESTINATION_AIRPORT', right_on = 'IATA_CODE', how='left')

flights = flights.rename(columns={
    'IATA_CODE': 'IATA_CODE_arr',
    'CITY': 'CITY_arr',
    'STATE': 'STATE_arr',
    'COUNTRY': 'COUNTRY_arr',
    'LATITUDE': 'LATITUDE_arr',
    'LONGITUDE': 'LONGITUDE_arr'
})

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


tail_numbers = tail_numbers[['Tail Number', 'Aircraft Model']]

