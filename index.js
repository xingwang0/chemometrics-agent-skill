/**
 * research-ml-modeling for DeepSeek Harness.
 *
 * Registers the bundled `research-ml-modeling` skill on `ctx.skills`. The skill
 * body lives in this package's `SKILL.md`; `references/`, `scripts/` and
 * `assets/` sit next to it, and the harness resolves the skill's base directory
 * at load time, so every relative path inside the skill keeps working wherever
 * the plugin is installed.
 *
 * The plugin imports nothing but Node builtins: a third-party plugin cannot
 * rely on `@deepseek-ai/*` resolving from its own directory, because those
 * packages are nested inside the dsh installation. Registries are reached
 * through `ctx` alone.
 *
 * @module dsh-research-ml-modeling
 */

import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

/** Cordis plugin name used by loader diagnostics. */
export const name = 'research-ml-modeling'

/** The registry this plugin registers into. */
export const inject = ['skills']

const packageRoot = dirname(fileURLToPath(import.meta.url))

/**
 * Split a `SKILL.md` into its frontmatter block and body.
 * @param content - the raw file text.
 * @returns the frontmatter text (empty when absent) and the trimmed body.
 */
function splitFrontmatter(content) {
  const match = content.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/)
  if (match === null) return { frontmatter: '', body: content.trim() }
  return { frontmatter: match[1], body: match[2].trim() }
}

/**
 * Remove the quoting a single-line YAML scalar may carry, without pulling in a
 * YAML parser. Escaped double quotes and backslashes are unescaped; anything
 * else is returned as written.
 * @param raw - the scalar text to the right of the colon.
 * @returns the unquoted value.
 */
function unquote(raw) {
  const text = raw.trim()
  if (text.length >= 2 && text.startsWith('"') && text.endsWith('"')) {
    return text.slice(1, -1).replace(/\\(["\\])/g, '$1')
  }
  if (text.length >= 2 && text.startsWith("'") && text.endsWith("'")) {
    return text.slice(1, -1).replace(/''/g, "'")
  }
  return text
}

/**
 * Read one-line scalar frontmatter keys. The bundled SKILL.md keeps `name`,
 * `description` and `version` on single lines, so block scalars are not needed.
 * @param frontmatter - the frontmatter text without its `---` delimiters.
 * @returns the parsed string values by key.
 */
function parseFields(frontmatter) {
  const fields = {}
  for (const line of frontmatter.split(/\r?\n/)) {
    const match = line.match(/^([A-Za-z][A-Za-z0-9_-]*):\s*(.*)$/)
    if (match !== null && match[2] !== '') fields[match[1]] = unquote(match[2])
  }
  return fields
}

/**
 * Register the bundled modelling skill.
 * @param ctx - plugin context; the registration is disposed with it.
 */
export function apply(ctx) {
  const skillFile = join(packageRoot, 'SKILL.md')
  let raw
  try {
    raw = readFileSync(skillFile, 'utf8')
  } catch (error) {
    // A missing SKILL.md only disables the skill, so warn and keep the plugin
    // loadable instead of failing the whole profile.
    ctx.logger.warn(
      `research-ml-modeling: cannot read SKILL.md under ${packageRoot}: ${
        error instanceof Error ? error.message : String(error)
      }`,
    )
    return
  }

  const { frontmatter, body } = splitFrontmatter(raw)
  const fields = frontmatter === '' ? {} : parseFields(frontmatter)
  if (fields.name === undefined || fields.description === undefined || body === '') {
    ctx.logger.warn(
      'research-ml-modeling: SKILL.md requires non-empty name, description and body; skill not registered',
    )
    return
  }

  ctx.skills.register({
    name: fields.name,
    description: fields.description,
    content: body,
    source: 'bundled',
    path: skillFile,
    resourceBase: { kind: 'directory', path: packageRoot },
  })
}
