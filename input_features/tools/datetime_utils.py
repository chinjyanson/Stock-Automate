import datetime

def get_current_date():
    return datetime.datetime.now().strftime("%Y-%m-%d")

def get_60days_ago():
    return (datetime.datetime.now() - datetime.timedelta(days=59)).strftime("%Y-%m-%d")

def time_difference_in_days(date1: str, date2: str):
    date1 = datetime.datetime.strptime(date1, "%Y-%m-%d")
    date2 = datetime.datetime.strptime(date2, "%Y-%m-%d")
    return (date2 - date1).days

if __name__ == "__main__":
    print(get_current_date())