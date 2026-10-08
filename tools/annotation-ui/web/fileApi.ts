import { createReadStream } from 'node:fs'
import fs from 'node:fs/promises'
import path from 'node:path'
import { timingSafeEqual } from 'node:crypto'
import { pipeline } from 'node:stream/promises'
import type { IncomingMessage, ServerResponse } from 'node:http'
import type { Plugin } from 'vite'
import { correctedAnnotationPath, detectVideoContext, overrunWarning } from '../electron/ipc/videoContext'

export interface WebApiOptions {
  setOpticsRoot: string
  videoRoot: string
  token: string
}

type Middleware = (req: IncomingMessage, res: ServerResponse, next: () => void) => Promise<void>
type Body = Record<string, unknown>
type Range = { start: number; end: number }

const VIDEO_EXTS = new Set(['.mp4', '.mov', '.webm'])
const MEDIA_TYPES: Record<string, string> = { '.mp4': 'video/mp4', '.mov': 'video/quicktime', '.webm': 'video/webm' }
const COOKIE = 'annot_key'
const MAX_BODY_BYTES = 10 * 1024 * 1024

class HttpError extends Error {
  constructor(readonly status: number, message: string) {
    super(message)
  }
}

function send(res: ServerResponse, status: number, body: unknown) {
  res.statusCode = status
  res.setHeader('Content-Type', 'application/json')
  res.end(JSON.stringify(body))
}

async function readBody(req: IncomingMessage): Promise<Body> {
  if (!(req.headers['content-type'] ?? '').startsWith('application/json')) throw new HttpError(415, 'expected application/json')
  const chunks: Buffer[] = []
  let size = 0
  for await (const chunk of req) {
    size += (chunk as Buffer).length
    if (size > MAX_BODY_BYTES) throw new HttpError(413, 'body too large')
    chunks.push(chunk as Buffer)
  }
  let body: unknown
  try {
    body = JSON.parse(Buffer.concat(chunks).toString('utf-8') || '{}')
  } catch {
    throw new HttpError(400, 'invalid JSON')
  }
  if (typeof body !== 'object' || body === null || Array.isArray(body)) throw new HttpError(400, 'expected a JSON object')
  return body as Body
}

function sameSecret(given: string, token: string): boolean {
  const a = Buffer.from(given)
  const b = Buffer.from(token)
  return a.length === b.length && timingSafeEqual(a, b)
}

function cookieToken(req: IncomingMessage): string | null {
  for (const part of (req.headers.cookie ?? '').split(';')) {
    const [name, ...value] = part.trim().split('=')
    if (name === COOKIE) return value.join('=')
  }
  return null
}

async function listVideos(root: string, depth = 2): Promise<{ path: string; name: string; sizeMb: number }[]> {
  let entries
  try {
    entries = await fs.readdir(root, { withFileTypes: true })
  } catch (error) {
    console.warn('[annotation-web-api] skipping unreadable directory', root, String(error))
    return []
  }
  const out: { path: string; name: string; sizeMb: number }[] = []
  for (const entry of entries) {
    const full = path.join(root, entry.name)
    if (entry.isDirectory() && depth > 0) {
      out.push(...(await listVideos(full, depth - 1)))
    } else if (entry.isFile() && VIDEO_EXTS.has(path.extname(entry.name).toLowerCase())) {
      const { size } = await fs.stat(full)
      out.push({ path: full, name: entry.name, sizeMb: Math.round(size / 1e6) })
    }
  }
  return out.sort((a, b) => a.path.localeCompare(b.path))
}

/** Parses a single-range `Range` header; null means serve the whole file. */
export function parseRange(header: string | undefined, size: number): Range | 'unsatisfiable' | null {
  const match = /^bytes=(\d*)-(\d*)$/.exec(header?.trim() ?? '')
  if (!match) return null
  const [, first, last] = match
  let start: number
  let end = size - 1
  if (first) {
    start = Number(first)
    if (last) end = Math.min(Number(last), size - 1)
  } else {
    if (!last || Number(last) === 0) return 'unsatisfiable'
    start = Math.max(0, size - Number(last))
  }
  return start >= size || start > end ? 'unsatisfiable' : { start, end }
}

async function isFile(p: string): Promise<boolean> {
  return fs.stat(p).then((s) => s.isFile(), () => false)
}

async function streamMedia(req: IncomingMessage, res: ServerResponse, filePath: string) {
  if (!(await isFile(filePath))) throw new HttpError(404, 'not a file')
  const { size } = await fs.stat(filePath)
  const range = parseRange(req.headers.range, size)
  if (range === 'unsatisfiable') {
    res.statusCode = 416
    res.setHeader('Content-Range', `bytes */${size}`)
    res.end()
    return
  }
  res.setHeader('Accept-Ranges', 'bytes')
  res.setHeader('Content-Type', MEDIA_TYPES[path.extname(filePath).toLowerCase()] ?? 'application/octet-stream')
  if (range) {
    res.statusCode = 206
    res.setHeader('Content-Range', `bytes ${range.start}-${range.end}/${size}`)
  }
  res.setHeader('Content-Length', range ? range.end - range.start + 1 : size)
  try {
    await pipeline(createReadStream(filePath, range ?? {}), res)
  } catch (error) {
    // Seeking aborts range requests constantly; pipeline has already closed the file.
    if ((error as NodeJS.ErrnoException).code !== 'ERR_STREAM_PREMATURE_CLOSE') throw error
  }
}

/** Resolves p if it lies strictly inside one of roots (symlinks resolved), else null. */
async function inside(roots: string[], p: unknown): Promise<string | null> {
  if (typeof p !== 'string' || !p) return null
  const resolved = path.resolve(p)
  const real = await fs.realpath(resolved).catch(async () =>
    path.join(await fs.realpath(path.dirname(resolved)).catch(() => path.dirname(resolved)), path.basename(resolved)))
  for (const root of roots) {
    const realRoot = await fs.realpath(root).catch(() => root)
    if (real.startsWith(realRoot + path.sep)) return resolved
  }
  return null
}

async function saveCorrected(writeRoots: string[], body: Body) {
  const videoPath = await inside(writeRoots, body.videoPath)
  if (!videoPath) throw new HttpError(403, 'path not allowed')
  if (!VIDEO_EXTS.has(path.extname(videoPath).toLowerCase()) || !(await isFile(videoPath))) {
    throw new HttpError(400, 'videoPath must be an existing video')
  }
  if (typeof body.jsonData !== 'string') throw new HttpError(400, 'jsonData must be a string')
  try {
    JSON.parse(body.jsonData)
  } catch {
    throw new HttpError(400, 'jsonData is not valid JSON')
  }
  const outPath = await inside(writeRoots, correctedAnnotationPath(videoPath))
  if (!outPath) throw new HttpError(403, 'path not allowed')
  await fs.copyFile(outPath, `${outPath}.backup`).catch((error: NodeJS.ErrnoException) => {
    if (error.code !== 'ENOENT') throw error
  })
  await fs.writeFile(outPath, body.jsonData, 'utf-8')
  return { success: true, path: outPath, warning: overrunWarning(body.jsonData) }
}

/** Connect middleware serving the file operations the Electron main process normally does. */
export function webApiMiddleware({ setOpticsRoot, videoRoot, token }: WebApiOptions): Middleware {
  const writeRoots = [path.resolve(videoRoot), path.resolve(setOpticsRoot, 'data')]
  const readRoots = [...writeRoots, path.resolve(setOpticsRoot, 'tools', 'annotation', 'annotations')]

  async function handle(req: IncomingMessage, res: ServerResponse, url: URL) {
    const route = url.pathname.slice('/web-api/'.length)
    if (route === 'videos' && req.method === 'GET') return send(res, 200, { videos: await listVideos(videoRoot) })
    if (route === 'media' && req.method === 'GET') {
      const filePath = await inside(readRoots, url.searchParams.get('path'))
      if (!filePath) throw new HttpError(403, 'path not allowed')
      return streamMedia(req, res, filePath)
    }
    if (req.method !== 'POST') throw new HttpError(404, 'not found')
    const body = await readBody(req)
    if (route === 'detect-context') {
      const videoPath = await inside(readRoots, body.videoPath)
      if (!videoPath) throw new HttpError(403, 'path not allowed')
      return send(res, 200, await detectVideoContext(videoPath, setOpticsRoot))
    }
    if (route === 'read') {
      const filePath = await inside(readRoots, body.filePath)
      if (!filePath) throw new HttpError(403, 'path not allowed')
      try {
        return send(res, 200, { success: true, data: JSON.parse(await fs.readFile(filePath, 'utf-8')) })
      } catch (error) {
        return send(res, 200, { success: false, error: String(error) })
      }
    }
    if (route === 'save-corrected') return send(res, 200, await saveCorrected(writeRoots, body))
    throw new HttpError(404, 'not found')
  }

  return async (req, res, next) => {
    const url = new URL(req.url ?? '/', 'http://local')
    const key = url.searchParams.get('key')
    const keyValid = key !== null && sameSecret(key, token)
    const cookie = cookieToken(req)
    if (!keyValid && !(cookie !== null && sameSecret(cookie, token))) {
      res.statusCode = 401
      res.end('Unauthorized. Open the link with ?key=<token>.')
      return
    }
    if (keyValid) {
      const secure = req.headers['x-forwarded-proto'] === 'https' ? '; Secure' : ''
      res.setHeader('Set-Cookie', `${COOKIE}=${token}; Path=/; HttpOnly; SameSite=Lax${secure}; Max-Age=2592000`)
      if (req.method === 'GET') {
        url.searchParams.delete('key')
        res.statusCode = 302
        res.setHeader('Location', url.pathname + url.search)
        res.end()
        return
      }
    }
    if (!url.pathname.startsWith('/web-api/')) return next()
    try {
      await handle(req, res, url)
    } catch (error) {
      if (res.headersSent) {
        console.error('[annotation-web-api]', error)
        res.destroy()
        return
      }
      if (error instanceof HttpError) return send(res, error.status, { success: false, error: error.message })
      console.error('[annotation-web-api]', error)
      send(res, 500, { success: false, error: String(error) })
    }
  }
}

/** Vite plugin wrapping webApiMiddleware, so the UI runs in a browser. */
export function annotationWebApi(options: WebApiOptions): Plugin {
  const middleware = webApiMiddleware(options)
  return {
    name: 'annotation-web-api',
    configureServer(server) {
      server.middlewares.use((req, res, next) => void middleware(req, res, next))
    },
  }
}
