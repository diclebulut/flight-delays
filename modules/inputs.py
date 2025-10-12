import pandas as pd
import modules.config as cfg

airlines = pd.read_csv(cfg.AIRLINES_DATA_PATH)
airports = pd.read_csv(cfg.AIRPORTS_DATA_PATH)
flights = pd.read_csv(cfg.FLIGHTS_DATA_PATH)
tail_numbers = pd.read_csv(cfg.TAIL_NUMBERS_DATA_PATH)


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

