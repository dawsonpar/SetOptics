import fs from 'node:fs/promises'
import http from 'node:http'
import os from 'node:os'
import path from 'node:path'
import type { AddressInfo } from 'node:net'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { parseRange, webApiMiddleware } from './fileApi'

const TOKEN = 'test-token-0123456789'
let tmp: string
let repo: string
let videos: string
let base: string
let server: http.Server

beforeAll(async () => {
  tmp = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'annot-web-')))
  repo = path.join(tmp, 'repo')
  videos = path.join(repo, 'data', 'rally-gt')
  await fs.mkdir(videos, { recursive: true })
  await fs.mkdir(path.join(repo, 'tools', 'annotation', 'annotations', 'game', 'rally'), { recursive: true })
  await fs.writeFile(path.join(videos, 'game.mp4'), '0123456789')
  await fs.writeFile(path.join(videos, 'notes.txt'), 'x')
  await fs.writeFile(path.join(repo, 'tools', 'annotation', 'annotations', 'game', 'rally', 'raw.json'), '{"a":1}')
  await fs.writeFile(path.join(tmp, 'secret.json'), '{}')
  const middleware = webApiMiddleware({ setOpticsRoot: repo, videoRoot: videos, token: TOKEN })
  server = http.createServer((req, res) => void middleware(req, res, () => {
    res.statusCode = 200
    res.end('page')
  }))
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`
})

afterAll(async () => {
  await new Promise((resolve) => server.close(resolve))
  await fs.rm(tmp, { recursive: true, force: true })
})

const auth = { Cookie: `annot_key=${TOKEN}` }

function post(route: string, body: unknown, headers: Record<string, string> = {}) {
  return fetch(`${base}/web-api/${route}`, {
    method: 'POST',
    headers: { ...auth, 'Content-Type': 'application/json', ...headers },
    body: JSON.stringify(body),
  })
}

function media(p: string, range?: string) {
  return fetch(`${base}/web-api/media?path=${encodeURIComponent(p)}`, { headers: { ...auth, ...(range ? { Range: range } : {}) } })
}

describe('parseRange', () => {
  it('clamps a suffix range larger than the file', () => {
    expect(parseRange('bytes=-500', 10)).toEqual({ start: 0, end: 9 })
  })
  it('rejects empty suffix ranges and starts past the end', () => {
    expect(parseRange('bytes=-0', 10)).toBe('unsatisfiable')
    expect(parseRange('bytes=-', 10)).toBe('unsatisfiable')
    expect(parseRange('bytes=10-', 10)).toBe('unsatisfiable')
  })
  it('serves the whole file for a missing or multi-range header', () => {
    expect(parseRange(undefined, 10)).toBeNull()
    expect(parseRange('bytes=0-1,4-5', 10)).toBeNull()
  })
})

describe('auth', () => {
  it('rejects requests without the token', async () => {
    expect((await fetch(`${base}/web-api/videos`)).status).toBe(401)
    expect((await fetch(`${base}/web-api/videos`, { headers: { Cookie: 'annot_key=wrong' } })).status).toBe(401)
  })
  it('sets the cookie and redirects away from ?key=', async () => {
    const res = await fetch(`${base}/editor?key=${TOKEN}&x=1`, { redirect: 'manual' })
    expect(res.status).toBe(302)
    expect(res.headers.get('location')).toBe('/editor?x=1')
    expect(res.headers.get('set-cookie')).toContain(`annot_key=${TOKEN}`)
    expect(res.headers.get('set-cookie')).not.toContain('Secure')
  })
})

describe('media', () => {
  it('refuses a root, returns 404 for a directory and keeps serving', async () => {
    expect((await media(path.join(repo, 'data'))).status).toBe(403)
    const dir = path.join(videos, 'dir')
    await fs.mkdir(dir)
    expect((await media(dir)).status).toBe(404)
    expect((await media(path.join(videos, 'game.mp4'))).status).toBe(200)
  })
  it('answers an oversized suffix range with the whole file', async () => {
    const res = await media(path.join(videos, 'game.mp4'), 'bytes=-500')
    expect(res.status).toBe(206)
    expect(res.headers.get('content-range')).toBe('bytes 0-9/10')
    expect(await res.text()).toBe('0123456789')
  })
  it('rejects an empty suffix range', async () => {
    expect((await media(path.join(videos, 'game.mp4'), 'bytes=-0')).status).toBe(416)
  })
  it('blocks paths outside the roots', async () => {
    expect((await media(path.join(videos, '..', '..', '..', 'secret.json'))).status).toBe(403)
  })
})

describe('request bodies', () => {
  it('rejects non-object JSON, bad JSON and other content types', async () => {
    expect((await post('read', null)).status).toBe(400)
    const bad = await fetch(`${base}/web-api/read`, { method: 'POST', headers: { ...auth, 'Content-Type': 'application/json' }, body: '{' })
    expect(bad.status).toBe(400)
    expect((await post('read', {}, { 'Content-Type': 'text/plain' })).status).toBe(415)
  })
})

describe('read', () => {
  it('reads raw drafts under tools/annotation/annotations', async () => {
    const res = await post('read', { filePath: path.join(repo, 'tools', 'annotation', 'annotations', 'game', 'rally', 'raw.json') })
    expect(await res.json()).toEqual({ success: true, data: { a: 1 } })
  })
})

describe('save-corrected', () => {
  const out = () => path.join(videos, 'game_annotations_corrected.json')

  it('refuses a root directory as the video', async () => {
    expect((await post('save-corrected', { videoPath: path.join(repo, 'data'), jsonData: '{}' })).status).toBe(403)
    await expect(fs.access(path.join(repo, 'data_annotations_corrected.json'))).rejects.toThrow()
  })
  it('refuses a path that is not an existing video', async () => {
    expect((await post('save-corrected', { videoPath: path.join(videos, 'notes.txt'), jsonData: '{}' })).status).toBe(400)
    expect((await post('save-corrected', { videoPath: path.join(videos, 'missing.mp4'), jsonData: '{}' })).status).toBe(400)
  })
  it('refuses invalid JSON', async () => {
    expect((await post('save-corrected', { videoPath: path.join(videos, 'game.mp4'), jsonData: '{' })).status).toBe(400)
  })
  it('writes beside the video and keeps the previous file as .backup', async () => {
    const first = await post('save-corrected', { videoPath: path.join(videos, 'game.mp4'), jsonData: '{"v":1}' })
    expect(await first.json()).toMatchObject({ success: true, path: out() })
    await post('save-corrected', { videoPath: path.join(videos, 'game.mp4'), jsonData: '{"v":2}' })
    expect(await fs.readFile(out(), 'utf-8')).toBe('{"v":2}')
    expect(await fs.readFile(`${out()}.backup`, 'utf-8')).toBe('{"v":1}')
  })
})
