from fundlens import FundClient

client = FundClient()

hits = client.search("HDFC Flexi Cap")
print(f"Found {len(hits)} schemes\n")

# Fetch data for all 3 funds
schemes_data = []
for h in hits[:3]:
    scheme = client.get_scheme(h.scheme_code)
    schemes_data.append({
        "scheme": scheme,
        "returns": scheme.returns(),
        "volatility": scheme.volatility(),
        "nav_points": len(scheme.nav_points),
    })

# Display each fund
for d in schemes_data:
    scheme = d["scheme"]
    print("=" * 60)
    print(f"  {scheme.scheme_name}")
    print(f"  Code     : {scheme.scheme_code}")
    print(f"  Latest NAV: ₹{scheme.latest_nav.nav:.3f} on {scheme.latest_nav.nav_date}")
    print(f"  Category : {scheme.meta.scheme_category}")
    print(f"  NAV history: {d['nav_points']} days of data")
    print(f"\n  Returns:")
    for period, value in d["returns"].items():
        print(f"    {period:>20}: {value * 100:+.2f}%")
    if d["volatility"]:
        print(f"\n  Volatility : {d['volatility'] * 100:.2f}%")
    print()

# Most invested = most NAV history (proxy for fund age/size; MFApi doesn't expose AUM)
most_invested = max(schemes_data, key=lambda d: d["nav_points"])
print("=" * 60)
print("Most invested fund (by trading history length):")
print(f"  {most_invested['scheme'].scheme_name}")
print(f"  {most_invested['nav_points']} days of NAV data")
print()
print("Note: MFApi.in does not provide AUM figures. NAV history length")
print("is used as a proxy — older/larger funds accumulate more data points.")
