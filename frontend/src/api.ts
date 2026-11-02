// Typed client for the fieldmark FastAPI backend (see backend/main.py).
// Vite's dev-server proxy (vite.config.ts) forwards /api/* to the backend,
// so relative paths work in both dev and a production same-origin deploy.

export interface LatLon {
  lat: number
  lon: number
}

export interface RegionData {
  lat0: number
  lon0: number
  half_width_km: number
  east_km: number[]
  north_km: number[]
  mag_grid: number[][]
  grav_grid: number[][]
}

export interface SimulateResponse {
  t: number[]
  true_lat: number[]
  true_lon: number[]
  est_lat: number[]
  est_lon: number[]
  pos_error_m: number[]
  pos_std_m: number[]
  est_heading_deg: number[]
  mag_updates: number
  grav_updates: number
  tail_rms_m: number
  final_error_m: number
  max_error_m: number
  region: RegionData
}

export interface SimulateParams {
  waypoints: LatLon[]
  imu_grade: string
  mag_noise_nt: number
  grav_noise_mgal: number
  use_mag: boolean
  use_grav: boolean
  speed_mps: number
  turn_rate_dps: number
  dt: number
  seed: number
  n_grid: number
}

export interface SweepParams {
  waypoints: LatLon[]
  imu_grade: string
  speed_mps: number
  turn_rate_dps: number
  dt: number
  n_trials: number
  n_grid: number
}

export interface SweepResponse {
  values: number[]
  means: number[]
  stds: number[]
}

export interface Sweep2DResponse {
  mag_values: number[]
  grav_values: number[]
  grid: number[][]
}

export interface DriftParams {
  waypoints: LatLon[]
  speed_mps: number
  turn_rate_dps: number
  dt: number
  n_trials: number
}

export interface DriftResponse {
  horizons_s: number[]
  by_grade: Record<string, number[]>
}

export class ApiError extends Error {}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => null)
    throw new ApiError(detail?.detail ?? `${path} failed (${res.status})`)
  }
  return res.json() as Promise<T>
}

export const api = {
  simulate: (params: SimulateParams) => post<SimulateResponse>('/api/simulate', params),
  sweepMag: (params: SweepParams) => post<SweepResponse>('/api/sweep/mag', params),
  sweepGrav: (params: SweepParams) => post<SweepResponse>('/api/sweep/grav', params),
  sweep2d: (params: SweepParams) => post<Sweep2DResponse>('/api/sweep/2d', params),
  drift: (params: DriftParams) => post<DriftResponse>('/api/drift', params),
}

export const IMU_GRADES = ['consumer_mems', 'tactical', 'navigation'] as const
export const NOISE_OPTIONS = [0.1, 0.3, 1, 3, 10, 30, 100]
