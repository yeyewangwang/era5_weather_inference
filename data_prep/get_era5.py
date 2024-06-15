import cdsapi, time, argparse, os


def retrieve(dest, year, month, date, hour, era_type):
    start_time = time.time()
    filename = os.path.join(dest, f"{month}_{year}_{date}_{hour}_{era_type}.nc")
    if era_type == "sfc":
        retrieve_sfc(filename, year, month, date, hour)
    elif era_type == "pl":
        retrieve_upper(filename, year, month, date, hour)

    elapsed_time = time.time() - start_time
    print(f"Success: Downloaded [{filename}] ... Time: [{elapsed_time:.5f} seconds]")
    return filename


def retrieve_upper(filename, year, month, date, hour):
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "geopotential",
                "specific_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
            ],
            "pressure_level": [
                "1000",
                "925",
                "850",
                "700",
                "600",
                "500",
                "400",
                "300",
                "250",
                "200",
                "150",
                "100",
                "50",
            ],
            "year": year,
            "month": month,
            "day": [
                date,
            ],
            "time": [
                f"{hour}:00",
            ],
        },
        filename,
    )


def retrieve_sfc(filename, year, month, date, hour):
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "mean_sea_level_pressure",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature",
            ],
            "year": year,
            "month": month,
            "day": [
                date,
            ],
            "time": [
                f"{hour}:00",
            ],
        },
        filename,
    )


def run_retrieve(dest, year, month, date, hour):
    filenames = []
    for era_type in ["sfc", "pl"]:
        filenames.append(retrieve(dest, year, month, date, hour, era_type))
    return filenames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=str, required=True)
    parser.add_argument("--year", type=str, required=True)
    parser.add_argument("--month", type=str, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--hour", type=str, required=True)
    args = parser.parse_args()

    run_retrieve(args.dest, args.year, args.month, args.date, args.hour)
