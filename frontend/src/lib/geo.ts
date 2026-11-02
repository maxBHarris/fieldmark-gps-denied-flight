// Small geo utilities -- no turf.js dependency needed for the one thing
// we use it for (a circle of given radius draped on the globe).

const EARTH_RADIUS_M = 6371008.7714

/** Destination point given a start lat/lon, bearing (deg), and distance (m). */
export function destinationPoint(lat: number, lon: number, bearingDeg: number, distanceM: number) {
  const angDist = distanceM / EARTH_RADIUS_M
  const bearing = (bearingDeg * Math.PI) / 180
  const lat1 = (lat * Math.PI) / 180
  const lon1 = (lon * Math.PI) / 180

  const lat2 = Math.asin(
    Math.sin(lat1) * Math.cos(angDist) + Math.cos(lat1) * Math.sin(angDist) * Math.cos(bearing),
  )
  const lon2 =
    lon1 +
    Math.atan2(
      Math.sin(bearing) * Math.sin(angDist) * Math.cos(lat1),
      Math.cos(angDist) - Math.sin(lat1) * Math.sin(lat2),
    )

  return [(lon2 * 180) / Math.PI, (lat2 * 180) / Math.PI] as [number, number] // [lng, lat], GeoJSON order
}

/**
 * A closed ring of [lng, lat] points forming a circle of `radiusM` meters
 * around (lat, lon) -- suitable for react-globe.gl's `polygonsData`
 * (GeoJSON-style `{ type: 'Polygon', coordinates: [ring] }`).
 */
export function circlePolygonRing(lat: number, lon: number, radiusM: number, numPoints = 64) {
  const ring: [number, number][] = []
  for (let i = 0; i <= numPoints; i++) {
    const bearing = (360 * i) / numPoints
    ring.push(destinationPoint(lat, lon, bearing, radiusM))
  }
  return ring
}

/**
 * Point offset `distanceM` perpendicular to `headingDeg`, to the right
 * (side=1) or left (side=-1) of travel. Used to build an uncertainty
 * corridor: two parallel offset lines, one on each side of a track.
 */
export function perpendicularOffset(
  lat: number,
  lon: number,
  headingDeg: number,
  distanceM: number,
  side: 1 | -1,
) {
  return destinationPoint(lat, lon, headingDeg + side * 90, distanceM)
}

/**
 * Rough client-side estimate of the field-map region half-width the backend
 * will size for these waypoints (mirrors backend/main.py's
 * _build_region_and_trajectory, using a simpler spherical approximation --
 * close enough for a live "here's roughly how big a region this needs"
 * hint, not the actual value used for the simulation).
 */
export function estimateRegionHalfWidthKm(
  waypoints: { lat: number; lon: number }[],
  speedMps: number,
  turnRateDps: number,
): number {
  if (waypoints.length === 0) return 0
  const lat0 = waypoints.reduce((s, w) => s + w.lat, 0) / waypoints.length
  const lon0 = waypoints.reduce((s, w) => s + w.lon, 0) / waypoints.length

  let halfExtentM = 0
  for (const w of waypoints) {
    const north = ((w.lat - lat0) * Math.PI) / 180 * EARTH_RADIUS_M
    const east = (((w.lon - lon0) * Math.PI) / 180) * EARTH_RADIUS_M * Math.cos((lat0 * Math.PI) / 180)
    halfExtentM = Math.max(halfExtentM, Math.abs(north), Math.abs(east))
  }

  const turnRadiusM = speedMps / Math.max((turnRateDps * Math.PI) / 180, 1e-6)
  const marginM = halfExtentM * 0.3 + 2.5 * turnRadiusM
  return Math.max(30, (halfExtentM + marginM) / 1000)
}
