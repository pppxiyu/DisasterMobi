"""
Track-derived hurricane exposure per city-event, from the NHC best track.

The registry's `ss_intensity` answers "what category was the STORM at its
closest approach".  For a city the storm passed 50 miles offshore that is not
what the city experienced, and the provenance note for those values says so
outright.  This module derives two alternatives from one authoritative source
(HURDAT2, the NHC's post-analysis best track), so the three x-axis definitions
in 5-disaster_vs_resi differ in what they measure, not in who compiled them:

    track_dist_km   the closest approach distance from the city centroid to the
                    (linearly interpolated) storm centre track.  Pure geometry:
                    no assumption about the wind field, and it separates cities
                    the category cannot — Ian's four are all "Cat 4" but sat
                    between the eyewall and 60 km inland.
    local_wind_kt   the wind the city actually sat in, read off the best track's
                    QUADRANT WIND RADII: at closest approach, if the city lies
                    inside the 64-kt radius for its quadrant it saw
                    hurricane-force wind, else 50-kt, else 34-kt, else below
                    tropical-storm force.  This is an operational analysis of
                    the wind FIELD rather than a station anemometer, which is
                    the price of having one uniform source for all 13 units —
                    a per-city station hunt would mix report types and coverage.

HURDAT2 records are 6-hourly (plus landfall specials).  A storm covers 100 km+
in six hours, so the track is interpolated to 15-minute steps before the
distance is minimised; the radii are taken from the bracketing record at the
closest-approach time, since radii are analysed per record and interpolating
them would invent a wind field.
"""
import io
import math

import numpy as np
import pandas as pd

# Basin file: nhc.noaa.gov/data/hurdat/  (Atlantic).  Downloaded once into
# data/hurdat2/; the parser tolerates any of the periodic re-releases.
HURDAT2_PATH = 'data/hurdat2/hurdat2_atlantic.txt'

# The quadrant order HURDAT2 writes for every radii group: NE, SE, SW, NW.
_QUADRANTS = ('NE', 'SE', 'SW', 'NW')
_EARTH_R_KM = 6371.0088
_NM_TO_KM = 1.852


def _parse_latlon(tok):
    v = float(tok[:-1])
    return -v if tok[-1] in 'SW' else v


def load_track(basin_id, path=HURDAT2_PATH):
    """One storm's best track as a DataFrame: time, lat, lon, status, wind_kt
    and the twelve radii columns (r34_NE ... r64_NW), in nautical miles.

    A radius of -999 means "not analysed" and 0 means "no wind of that strength
    in that quadrant"; both become NaN and 0.0 respectively so a missing
    analysis is never read as a small radius.
    """
    lines = io.open(path, encoding='utf-8').read().splitlines()
    i, rows = 0, None
    while i < len(lines):
        p = [x.strip() for x in lines[i].split(',')]
        if len(p) >= 3 and p[0].startswith('AL') and p[2].isdigit():
            n = int(p[2])
            if p[0] == basin_id:
                rows = [[x.strip() for x in lines[j].split(',')]
                        for j in range(i + 1, i + 1 + n)]
                break
            i += 1 + n
        else:
            i += 1
    if rows is None:
        raise KeyError(f'{basin_id} not found in {path}')

    rec = []
    for r in rows:
        radii = []
        for v in r[8:20]:
            x = float(v)
            radii.append(np.nan if x < 0 else x)
        rec.append(dict(
            time=pd.to_datetime(r[0] + r[1], format='%Y%m%d%H%M'),
            status=r[3], lat=_parse_latlon(r[4]), lon=_parse_latlon(r[5]),
            wind_kt=float(r[6]),
            **{f'r{t}_{q}': radii[k * 4 + j]
               for k, t in enumerate((34, 50, 64))
               for j, q in enumerate(_QUADRANTS)}))
    return pd.DataFrame(rec)


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * _EARTH_R_KM * np.arcsin(np.sqrt(a))


def _quadrant(storm_lat, storm_lon, city_lat, city_lon):
    """Which storm-relative quadrant the city sits in (radii are per quadrant)."""
    north = city_lat >= storm_lat
    east = city_lon >= storm_lon
    return 'NE' if (north and east) else 'SE' if east else 'NW' if north else 'SW'


def city_exposure(track, city_lat, city_lon, window=None, step_min=15):
    """Closest approach of `track` to one city, and the wind field there.

    `window` optionally restricts the search to (start, end) timestamps, so a
    looping storm cannot be matched at an irrelevant pass.  Returns a dict:
      dist_km        minimum centre-to-city distance
      time           when that happened (interpolated)
      storm_wind_kt  the storm's own intensity at that moment (interpolated)
      quadrant       the city's storm-relative quadrant
      local_wind_kt  64 / 50 / 34 / 0 — the strongest radius band the city
                     falls inside, i.e. the wind it sat in
      band           a readable label for local_wind_kt
    """
    t = track if window is None else track[(track.time >= window[0]) &
                                           (track.time <= window[1])]
    if len(t) < 2:
        t = track
    # Interpolate the centre to `step_min` so a 6-hourly record cannot put the
    # closest approach 100 km off.
    ts = t.time.astype('int64').to_numpy()
    grid = np.arange(ts[0], ts[-1], step_min * 60 * 1_000_000_000)
    lat = np.interp(grid, ts, t.lat.to_numpy())
    lon = np.interp(grid, ts, t.lon.to_numpy())
    wind = np.interp(grid, ts, t.wind_kt.to_numpy())
    d = _haversine_km(lat, lon, city_lat, city_lon)
    k = int(np.argmin(d))

    # Radii come from the NEAREST ACTUAL record, not an interpolation: they are
    # analysed per record and a blended radius would describe no real wind field.
    j = int(np.argmin(np.abs(ts - grid[k])))
    row = t.iloc[j]
    quad = _quadrant(lat[k], lon[k], city_lat, city_lon)
    local, band = 0.0, 'below TS force'
    for thr, lbl in ((64, 'hurricane force (>=64 kt)'),
                     (50, 'strong TS force (>=50 kt)'),
                     (34, 'TS force (>=34 kt)')):
        r_nm = row.get(f'r{thr}_{quad}', np.nan)
        if np.isfinite(r_nm) and r_nm > 0 and d[k] <= r_nm * _NM_TO_KM:
            local, band = float(thr), lbl
            break
    return dict(dist_km=float(d[k]),
                time=pd.Timestamp(grid[k]),
                storm_wind_kt=float(wind[k]),
                quadrant=quad, local_wind_kt=local, band=band)


def wind_field(track, lats, lons):
    """Peak modelled wind (kt) at each point, maximised over the whole track.

    `city_exposure` answers the same question for ONE point and returns a BAND
    (64 / 50 / 34 / 0), which is what the 5-disaster_vs_resi x-axis needs but
    is useless inside a city: every block group of a 30-km study area lands in
    the same band, so a banded column is constant within the unit and any
    within-city standardization sends it to zero.  This returns a CONTINUOUS
    value instead, which is what a component-level exposure has to be built
    from.

    The profile between the analysed radii is a modified Rankine vortex, the
    standard parametric choice when only the 34/50/64-kt radii are available:

        d <= r64          64 + (vmax - 64) * (1 - d / r64)
        r64 < d <= r50    linear from 64 down to 50
        r50 < d <= r34    linear from 50 down to 34
        d  > r34          34 * (r34 / d) ** 0.6

    The exponent 0.6 is the conventional outer decay rate; it only affects
    points beyond tropical-storm force, where every study area's block groups
    sit on the same side of the gradient, so the within-city ORDERING it
    produces is insensitive to the choice.

    Quadrant selection reuses `_quadrant`, so this function and `city_exposure`
    read the same radius for the same geometry.  HURDAT2 quadrants are compass
    quadrants about the centre (NE spans due north clockwise to due east),
    which is exactly the latitude/longitude sign test `_quadrant` applies.

    The maximum is taken over every record rather than at closest approach,
    because a slow-moving storm can deliver its strongest wind to a block group
    well before or after the centre's nearest pass.
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    best = np.zeros(len(lats), dtype=float)
    for row in track.itertuples(index=False):
        d = _haversine_km(row.lat, row.lon, lats, lons)
        north, east = lats >= row.lat, lons >= row.lon
        # index into _QUADRANTS = (NE, SE, SW, NW)
        quad = np.where(north & east, 0, np.where(east, 1, np.where(north, 3, 2)))
        r = {}
        for thr in (34, 50, 64):
            per_quad = np.array([getattr(row, f'r{thr}_{q}') for q in _QUADRANTS],
                                dtype=float) * _NM_TO_KM
            r[thr] = per_quad[quad]
        r34, r50, r64 = r[34], r[50], r[64]
        vmax = float(row.wind_kt)

        # Inner anchor per branch.  A radius of 0 means the record states there
        # is NO wind of that strength anywhere in the quadrant, so interpolating
        # DOWN from 64 kt there would invent hurricane-force wind at the eye of a
        # storm the record says never reached it (measured on these 13 units:
        # 414 of Baton Rouge's 500 block groups moved, mean 7.0 kt, max 14.6 kt).
        # Where the inner radius is absent the profile therefore starts from the
        # record's own vmax instead.
        v = np.full(len(lats), np.nan)
        m = np.isfinite(r64) & (r64 > 0) & (d <= r64)
        v[m] = 64 + (vmax - 64) * (1 - d[m] / r64[m])
        m = (np.isfinite(r64) & np.isfinite(r50) & (r50 > r64)
             & (d > r64) & (d <= r50))
        inner = np.where(r64 > 0, 64.0, vmax)
        v[m] = inner[m] + (50 - inner[m]) * (d[m] - r64[m]) / (r50[m] - r64[m])
        m = (np.isfinite(r50) & np.isfinite(r34) & (r34 > r50)
             & (d > r50) & (d <= r34))
        inner = np.where(r50 > 0, 50.0, np.where(r64 > 0, 64.0, vmax))
        v[m] = inner[m] + (34 - inner[m]) * (d[m] - r50[m]) / (r34[m] - r50[m])
        m = np.isfinite(r34) & (r34 > 0) & (d > r34)
        v[m] = 34.0 * (r34[m] / d[m]) ** 0.6
        best = np.fmax(best, np.nan_to_num(v))
    return best


def track_distance(track, lats, lons, step_min=15):
    """Minimum centre-to-point great-circle distance (km) over the whole track.

    Pure geometry: the best track's centre positions and nothing else -- no wind
    field, no radii, no vortex profile.  That is the point of it.  Every
    modelled-exposure definition tried on these 13 units (parametric wind under
    eight defensible variants, MRMS storm rainfall) either failed to beat this or
    beat it only inside the spread its own implementation choices could produce.

    The track is interpolated to `step_min` first, for the reason city_exposure
    interpolates: a 6-hourly record can leave the closest approach 100 km off.
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    ts = track.time.astype('int64').to_numpy()
    grid = np.arange(ts[0], ts[-1], step_min * 60 * 1_000_000_000)
    tlat = np.interp(grid, ts, track.lat.to_numpy())
    tlon = np.interp(grid, ts, track.lon.to_numpy())
    best = np.full(len(lats), np.inf)
    for la, lo in zip(tlat, tlon):
        best = np.minimum(best, _haversine_km(la, lo, lats, lons))
    return best


def landfall_day_wind(track, lats, lons, when):
    """City-wide MEAN modelled wind (kt) on the day of `when`.

    The per-event severity the rank channel's interaction term is moderated by.
    Records are restricted to the calendar day (UTC) of the city's own closest
    approach -- the same storm reaches Slidell and Deltona 36 hours apart, so a
    fixed calendar day would compare different phases of the storm.  Falls back
    to the single nearest record when no record shares that day.
    """
    day = pd.Timestamp(when).normalize()
    same = track[track.time.dt.normalize() == day]
    if len(same) == 0:
        j = int(np.argmin(np.abs(track.time - pd.Timestamp(when))))
        same = track.iloc[[j]]
    return float(np.mean(wind_field(same, lats, lons)))
