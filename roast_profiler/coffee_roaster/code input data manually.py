from roast_profiler.input_data import load_csv, validate_roast_data

df = load_csv('roast.csv')
validation = validate_roast_data(df)

