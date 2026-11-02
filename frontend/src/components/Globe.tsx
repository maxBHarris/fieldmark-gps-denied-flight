import { useEffect, useMemo, useRef, useState } from 'react'
import GlobeGL from 'react-globe.gl'
import type { LatLon, SimulateResponse } from '../api'
import { perpendicularOffset } from '../lib/geo'

const GLOBE_IMAGE_URL = '//unpkg.com/three-globe/example/img/earth-blue-marble.jpg'
const BUMP_IMAGE_URL = '//unpkg.com/three-globe/example/img/earth-topology.png'
const BACKGROUND_IMAGE_URL = '//unpkg.com/three-globe/example/img/night-sky.png'

interface GlobeViewProps {
  waypoints: LatLon[]
  planningMode: boolean
  onAddWaypoint: (coords: LatLon) => void
  simResult: SimulateResponse | null
  currentIndex: number
  sigmaMultiplier: number
}

type PointDatum = {
  lat: number
  lng: number
  kind: 'waypoint' | 'true' | 'est'
  label: string
}

const POINT_COLOR: Record<PointDatum['kind'], string> = {
  waypoint: '#9ca3af',
  true: '#38bdf8',
  est: '#f87171',
}

const POINT_RADIUS: Record<PointDatum['kind'], number> = {
  waypoint: 0.35,
  true: 0.55,
  est: 0.55,
}

function Legend({ showResult }: { showResult: boolean }) {
  const items: { color: string; label: string; dashed?: boolean; swatch?: 'line' | 'dot' | 'fill' }[] = [
    { color: '#9ca3af', label: 'Planned route (waypoints)', dashed: true, swatch: 'line' },
  ]
  if (showResult) {
    items.push(
      { color: '#38bdf8', label: 'True flight path', swatch: 'line' },
      { color: 'rgba(248, 113, 113, 0.45)', label: 'Predicted-location corridor (EKF)', swatch: 'fill' },
      { color: '#f87171', label: 'EKF estimate (at scrub time)', swatch: 'dot' },
      { color: '#38bdf8', label: 'True position (at scrub time)', swatch: 'dot' },
    )
  }
  return (
    <div className="pointer-events-none absolute bottom-3 left-3 space-y-1 rounded-lg bg-slate-950/70 p-3 text-xs text-slate-200 backdrop-blur">
      {items.map((item) => (
        <div key={item.label} className="flex items-center gap-2">
          {item.swatch === 'dot' && (
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: item.color }} />
          )}
          {item.swatch === 'fill' && (
            <span className="h-2.5 w-4 rounded-sm" style={{ backgroundColor: item.color }} />
          )}
          {item.swatch === 'line' && (
            <span
              className="h-0.5 w-4"
              style={{
                backgroundColor: item.dashed ? 'transparent' : item.color,
                borderTop: item.dashed ? `2px dashed ${item.color}` : undefined,
              }}
            />
          )}
          <span>{item.label}</span>
        </div>
      ))}
    </div>
  )
}

export default function GlobeView({
  waypoints,
  planningMode,
  onAddWaypoint,
  simResult,
  currentIndex,
  sigmaMultiplier,
}: GlobeViewProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ width: 600, height: 600 })

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect
      setSize({ width, height })
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  const points = useMemo<PointDatum[]>(() => {
    const pts: PointDatum[] = waypoints.map((w, i) => ({
      lat: w.lat,
      lng: w.lon,
      kind: 'waypoint',
      label: `Waypoint ${i + 1}`,
    }))
    if (simResult) {
      pts.push({
        lat: simResult.true_lat[currentIndex],
        lng: simResult.true_lon[currentIndex],
        kind: 'true',
        label: `True position (t=${simResult.t[currentIndex].toFixed(0)}s)`,
      })
      pts.push({
        lat: simResult.est_lat[currentIndex],
        lng: simResult.est_lon[currentIndex],
        kind: 'est',
        label: `EKF estimate (t=${simResult.t[currentIndex].toFixed(0)}s, error ${simResult.pos_error_m[currentIndex].toFixed(0)}m)`,
      })
    }
    return pts
  }, [waypoints, simResult, currentIndex])

  const paths = useMemo(() => {
    const result: { points: [number, number][]; color: string; dash: boolean }[] = []
    if (waypoints.length >= 2) {
      result.push({
        points: waypoints.map((w) => [w.lat, w.lon]),
        color: '#6b7280',
        dash: true,
      })
    }
    if (simResult) {
      // Full flown track, always shown -- not clipped to the scrub position,
      // so the whole route is visible as soon as a simulation finishes.
      const flown: [number, number][] = simResult.true_lat.map((lat, i) => [lat, simResult.true_lon[i]])
      result.push({ points: flown, color: '#38bdf8', dash: false })
    }
    return result
  }, [waypoints, simResult])

  // A continuous corridor along the EKF's estimated track, width = k-sigma
  // of its own reported position uncertainty at each point -- "where the
  // system thinks the platform is" over the whole route, not just one
  // instant. Built as a ribbon polygon: forward along the left offset,
  // backward along the right offset.
  const corridorPolygon = useMemo(() => {
    if (!simResult) return []
    const left: [number, number][] = []
    const right: [number, number][] = []
    for (let i = 0; i < simResult.t.length; i++) {
      const radiusM = Math.max(30, sigmaMultiplier * simResult.pos_std_m[i])
      const heading = simResult.est_heading_deg[i]
      left.push(perpendicularOffset(simResult.est_lat[i], simResult.est_lon[i], heading, radiusM, 1))
      right.push(perpendicularOffset(simResult.est_lat[i], simResult.est_lon[i], heading, radiusM, -1))
    }
    const ring = [...left, ...right.reverse()]
    if (ring.length) ring.push(ring[0])
    return [{ geometry: { type: 'Polygon', coordinates: [ring] } }]
  }, [simResult, sigmaMultiplier])

  return (
    <div
      ref={containerRef}
      className={`relative h-full w-full overflow-hidden rounded-xl ${planningMode ? 'cursor-crosshair' : ''}`}
    >
      <GlobeGL
        width={size.width}
        height={size.height}
        globeImageUrl={GLOBE_IMAGE_URL}
        bumpImageUrl={BUMP_IMAGE_URL}
        backgroundImageUrl={BACKGROUND_IMAGE_URL}
        showAtmosphere
        atmosphereColor="#60a5fa"
        onGlobeClick={(coords) => {
          if (planningMode) onAddWaypoint({ lat: coords.lat, lon: coords.lng })
        }}
        pointsData={points}
        pointLat="lat"
        pointLng="lng"
        pointColor={(d) => POINT_COLOR[(d as PointDatum).kind]}
        pointRadius={(d) => POINT_RADIUS[(d as PointDatum).kind]}
        pointAltitude={0.014}
        pointLabel={(d) => (d as PointDatum).label}
        pathsData={paths}
        pathPoints="points"
        pathPointLat={(p: unknown) => (p as [number, number])[0]}
        pathPointLng={(p: unknown) => (p as [number, number])[1]}
        pathColor={(d: unknown) => (d as (typeof paths)[number]).color}
        pathDashLength={(d: unknown) => ((d as (typeof paths)[number]).dash ? 0.01 : 1)}
        pathDashGap={(d: unknown) => ((d as (typeof paths)[number]).dash ? 0.008 : 0)}
        pathStroke={1.5}
        polygonsData={corridorPolygon}
        polygonGeoJsonGeometry="geometry"
        polygonCapColor={() => 'rgba(248, 113, 113, 0.35)'}
        polygonSideColor={() => 'rgba(248, 113, 113, 0.2)'}
        polygonStrokeColor={() => 'rgba(248, 113, 113, 0.6)'}
        polygonAltitude={0.006}
        polygonLabel={() => `Predicted-location corridor (${sigmaMultiplier}σ)`}
      />
      <Legend showResult={!!simResult} />
    </div>
  )
}
