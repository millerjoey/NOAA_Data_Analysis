import urllib.request, urllib.error

# Format of data:
# 0:12 is USAF and WBAN.
# 43:50 goes like "US____TX"
# 82:86 Is start year
# 91:95 is End year

def get_stations(state, first_year, last_year):
    src = urllib.request.urlopen("ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-history.txt")
    out = []
    for line in src:
        try:
            if (line[43:50].decode("utf-8") == "US   " + state) & (int(line[82:86].decode("utf-8")) < first_year) & (int(line[91:95].decode("utf-8")) > last_year):
                out.append(line[0:12].decode("utf-8").replace(" ", "-"))
        except ValueError:
            continue
    src.close()
    return(out)

def download_files(stations_list, first_year, last_year, path = "data/"):
    for yr in range(first_year, last_year + 1):
        url = "ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-lite/" + str(yr)
        for station in stations_list:
            try:
                urllib.request.urlretrieve(url + "/" + station + "-" + str(yr) + ".gz", path + station + "-" + str(yr) + ".gz")
            except urllib.error.URLError:
                print("Couldn't find file: ..." + station + "-" + str(yr) + ".gz. Skipping.")
                continue

    