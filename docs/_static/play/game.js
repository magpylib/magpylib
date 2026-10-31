/*
 * Super Magrio: drawing, input and sound on top of sim.js.
 *
 * Canvas pixels are 36 to the tile at zoom 1, in a 960 x 540 view. The background
 * glow and the field lines are computed once from the same formulas the hero
 * feels, so what is drawn is what pushes.
 */
(() => {
  "use strict";

  const Sim = globalThis.MagrioSim, Field = globalThis.MagField;
  const { W, H, GROUND, TOP, LEVEL_W, MAGNETS, SOLIDS, CHECKPOINTS } = Sim;

  const T = 36, VW = 960, VH = 540, Z_MAX = 26;
  const RED = "#aa0000", INK = "#2a2939", PAPER = "#f6f3ea", BLUE = "#2e91e5", PUSH = "#1f74c4";
  const NORTH = "#e71111", MIDDLE = "#dddddd", SOUTH = "#00b050", GOLD = "#d9a441";
  /* the logo's own proportions, in the pixels of magpylib_logo.png */
  const LOGO = { cx: 160.5, cz: 198.5, s: H / 283, arch: 173, outer: 115.5, inner: 59.5 };
  const BIRD_BOX = { x: 0, y: 0, w: 117, h: 188 }; /* magpylib_logo_bird.png sits at the top left of the logo */
  const FRAME = { w: 225, h: 210, n: 6 };
  const WON_WAIT = 1.1; /* seconds of the M flying home before the ending shows */

  const canvas = document.getElementById("game");
  const ctx = canvas.getContext("2d");
  const touchUI = matchMedia("(pointer: coarse)").matches;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  const birdStill = new Image(), birdFly = new Image();
  birdStill.src = "../images/magpylib_logo_bird.png";
  birdFly.src = "../images/magpylib_logo_bird_fly.png";

  let s = Sim.create();
  const input = { left: false, right: false, jump: false, flip: false };
  /* a press stays noted until a step has read it, however short the tap and slow the frame */
  const tapped = { jump: false, flip: false };
  const view = { x: 13, z: 0.6, zoom: 2.1 };
  let particles = [], clock = 0, winT = 0;
  let hintText = "", hintShown = "", hintFade = 0;

  /* ---------- what the field looks like, computed once ---------- */

  const HEAT_RES = 2; /* samples per tile: stretched and smoothed when drawn */
  const heat = document.createElement("canvas");
  function bakeHeat() {
    heat.width = (LEVEL_W + 8) * HEAT_RES;
    heat.height = Z_MAX * HEAT_RES;
    const g = heat.getContext("2d");
    const img = g.createImageData(heat.width, heat.height);
    for (let j = 0, k = 0; j < heat.height; j++) {
      const z = Z_MAX - (j + 0.5) / HEAT_RES;
      for (let i = 0; i < heat.width; i++, k += 4) {
        const b = Sim.absB(i / HEAT_RES - 4, z);
        const t = Math.max(0, Math.min(1, (Math.log10(b + 1e-9) + 2.6) / 2.4));
        img.data[k] = 46;
        img.data[k + 1] = 145;
        img.data[k + 2] = 229;
        img.data[k + 3] = 150 * t * t;
      }
    }
    g.putImageData(img, 0, 0);
  }

  const out = [0, 0, 0];
  function B2(x, z) {
    out[0] = out[1] = out[2] = 0;
    for (const m of MAGNETS) {
      const half = m.kind === "cuboid" ? m.dim[0] / 2 : 3;
      if (Math.abs(x - m.x) < 40 + half) Field.addB(m, x, 0, z, out);
    }
    return out;
  }

  function insideMagnet(x, z) {
    for (const m of MAGNETS) {
      if (m.kind === "sphere") {
        if (Math.hypot(x - m.x, z - m.z) < m.dim / 2) return true;
      } else {
        const [w, h] = m.kind === "cuboid" ? [m.dim[0], m.dim[2]] : m.dim;
        if (Math.abs(x - m.x) < w / 2 && Math.abs(z - m.z) < h / 2) return true;
      }
    }
    return false;
  }

  /* follows B from a point just outside a north face until it lands on something */
  function trace(x, z) {
    const h = 0.2, pts = [[x, z]];
    for (let i = 0; i < 900; i++) {
      let b = B2(x, z);
      let n = Math.hypot(b[0], b[2]) || 1;
      const mx = x + (0.5 * h * b[0]) / n, mz = z + (0.5 * h * b[2]) / n;
      b = B2(mx, mz);
      n = Math.hypot(b[0], b[2]) || 1;
      x += (h * b[0]) / n;
      z += (h * b[2]) / n;
      pts.push([x, z]);
      if (z < -1 || z > Z_MAX || insideMagnet(x, z)) break;
    }
    return pts;
  }

  function seeds(m) {
    const up = m.J[2] > 0 ? 1 : -1, list = [];
    if (m.kind === "sphere") {
      for (const deg of [-60, -36, -14, 14, 36, 60]) {
        const a = (deg * Math.PI) / 180, r = m.dim / 2 + 0.03;
        list.push([m.x + r * Math.sin(a), m.z + up * r * Math.cos(a)]);
      }
      return list;
    }
    const [w, h] = m.kind === "cuboid" ? [m.dim[0], m.dim[2]] : m.dim;
    const n = Math.min(14, Math.max(5, Math.round(w / 1.1)));
    for (let i = 0; i < n; i++) {
      list.push([m.x + (w * (i + 0.5)) / n - w / 2, m.z + up * (h / 2 + 0.03)]);
    }
    return list;
  }

  const lines = [];
  function bakeLines() {
    for (const m of MAGNETS) {
      const path = new Path2D();
      let x0 = Infinity, x1 = -Infinity;
      for (const [sx, sz] of seeds(m)) {
        const pts = trace(sx, sz);
        pts.forEach(([x, z], i) => {
          if (i) path.lineTo(x * T, -z * T);
          else path.moveTo(x * T, -z * T);
          x0 = Math.min(x0, x);
          x1 = Math.max(x1, x);
        });
      }
      lines.push({ path, x0, x1 });
    }
  }

  /* ---------- sound: small synthesized blips, nothing to download ---------- */

  let audio = null, muted = false;
  function blip(freq, to, dur, type = "square", vol = 0.05) {
    if (muted || !audio) return;
    const t = audio.currentTime, o = audio.createOscillator(), g = audio.createGain();
    o.type = type;
    o.frequency.setValueAtTime(freq, t);
    o.frequency.exponentialRampToValueAtTime(to, t + dur);
    g.gain.setValueAtTime(vol, t);
    g.gain.exponentialRampToValueAtTime(0.0008, t + dur);
    o.connect(g).connect(audio.destination);
    o.start(t);
    o.stop(t + dur);
  }

  const SOUND = {
    jump: () => blip(320, 620, 0.14),
    sensor: () => { blip(990, 990, 0.06); setTimeout(() => blip(1320, 1320, 0.12), 60); },
    flip: () => s.p.pol > 0 ? blip(760, 240, 0.16, "sawtooth", 0.03) : blip(240, 760, 0.16, "sawtooth", 0.03),
    stomp: () => blip(180, 70, 0.14),
    hurt: () => blip(420, 90, 0.34, "triangle", 0.09),
    squawk: () => { blip(1250, 700, 0.09, "sawtooth", 0.025); setTimeout(() => blip(1100, 620, 0.12, "sawtooth", 0.025), 110); },
    pluck: () => blip(880, 1400, 0.05, "triangle", 0.04),
    check: () => blip(520, 780, 0.2, "triangle", 0.06),
    clink: () => blip(1700, 1500, 0.08, "triangle", 0.05),
    win: () => [523, 659, 784, 1047].forEach((f, i) => setTimeout(() => blip(f, f, 0.2, "triangle", 0.08), i * 110)),
  };

  /* ---------- the hero: the logo's magnet, on its pole tips ---------- */

  function rect(x0, y0, x1, y1) {
    ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
  }

  function drawMagnetBody(color, liftL, liftR, eyes) {
    const { arch, outer, inner, cx } = LOGO;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(cx, arch, outer, Math.PI, 0);
    ctx.arc(cx, arch, inner, 0, Math.PI, true);
    ctx.closePath();
    ctx.fill();
    /* the legs shorten as the feet lift, the white gap above each foot stays */
    rect(cx - outer, arch - 1, cx - inner, 264 - liftL);
    rect(cx + inner, arch - 1, cx + outer, 264 - liftR);
    rect(cx - outer, 284 - liftL, cx - inner, 340 - liftL);
    rect(cx + inner, 284 - liftR, cx + outer, 340 - liftR);

    if (eyes > 0) {
      const look = s.p.face * 7, blink = (clock % 3.7) < 0.1 ? 0.15 : 1;
      for (const ex of [cx - 30, cx + 30]) {
        ctx.fillStyle = "#fff";
        ctx.beginPath();
        ctx.ellipse(ex, 86, 19, 19 * eyes * blink, 0, 0, 7);
        ctx.fill();
        ctx.fillStyle = INK;
        ctx.beginPath();
        ctx.ellipse(ex + look, 87, 10, 10 * eyes * blink, 0, 0, 7);
        ctx.fill();
      }
    }
  }

  function drawHero() {
    const p = s.p;
    if (p.safe && Math.floor(clock * 14) % 2) return;
    const k = LOGO.s * T;
    ctx.save();
    ctx.translate(p.x * T, -(p.z + H / 2) * T);
    ctx.rotate(p.angle);
    const flip = p.flipT > 0 ? Math.abs(Math.cos((Math.PI * p.flipT) / 0.28)) : 1;
    const squash = p.land / 0.16;
    ctx.scale(k * Math.max(0.12, flip) * (1 + 0.12 * squash), k * (1 - 0.14 * squash));
    ctx.translate(-LOGO.cx, -LOGO.cz);
    const moving = p.grounded || p.stuck, step = Math.sin(p.walk);
    let liftL = 0, liftR = 0;
    if (s.phase === "play" || s.phase === "won") {
      if (!moving) liftL = liftR = 24;
      else if (Math.abs(p.vx) + Math.abs(p.vz) > 0.8) {
        liftL = Math.max(0, step) * 34;
        liftR = Math.max(0, -step) * 34;
      }
    }
    const eyes = s.phase === "title" ? 0 : Math.min(1, s.t * 3);
    drawMagnetBody(p.pol > 0 ? RED : PUSH, liftL, liftR, eyes);
    if (s.bird.mode === "logo" && birdStill.complete) {
      ctx.drawImage(birdStill, BIRD_BOX.x, BIRD_BOX.y);
    }
    ctx.restore();

    /* the force the hero feels right now, the thing the game teaches */
    const a = Math.hypot(p.a.x, p.a.z), g = a / 38;
    if (g > 0.15 && s.phase === "play") {
      const len = Math.min(3, 0.45 + g * 0.5) * T, ux = p.a.x / a, uz = p.a.z / a;
      const x0 = p.x * T, y0 = -(p.z + H / 2) * T, x1 = x0 + ux * len, y1 = y0 - uz * len;
      ctx.strokeStyle = ctx.fillStyle = p.pol > 0 ? RED : PUSH;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x1 + ux * 8, y1 - uz * 8);
      ctx.lineTo(x1 - uz * 6, y1 - ux * 6);
      ctx.lineTo(x1 + uz * 6, y1 + ux * 6);
      ctx.fill();
    }
  }

  /* ---------- the magpie ---------- */

  function drawBird() {
    const b = s.bird;
    if (b.mode === "logo" || b.mode === "gone") return;
    const k = LOGO.s * T;
    ctx.save();
    ctx.translate(b.x * T, -b.z * T);
    if (b.mode === "perched") {
      /* looking back at the hero, tail hanging over the edge it stands on */
      ctx.scale(-k, k);
      if (birdStill.complete) ctx.drawImage(birdStill, -BIRD_BOX.w * 0.55, -BIRD_BOX.h * 0.62);
    } else {
      /* the sheet is drawn flying left */
      ctx.scale(b.face > 0 ? -k : k, k);
      const f = Math.floor(b.flap * 16) % FRAME.n;
      if (birdFly.complete) {
        ctx.drawImage(birdFly, f * FRAME.w, 0, FRAME.w, FRAME.h, -FRAME.w / 2, -FRAME.h / 2, FRAME.w, FRAME.h);
      }
    }
    ctx.restore();
  }

  function drawLetters() {
    const b = s.bird;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = "800 34px system-ui, -apple-system, 'Segoe UI', sans-serif";
    if (s.rope.length) {
      ctx.strokeStyle = INK;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      const [ax, az] = Sim.ropeAnchor(b);
      ctx.moveTo(ax * T, -az * T);
      for (const l of s.rope) ctx.lineTo(l.x * T, -l.z * T);
      ctx.stroke();
    }
    ctx.fillStyle = INK;
    for (const l of s.letters) {
      if (s.drop && l.ch === s.drop.ch) continue;
      ctx.fillText(l.ch, l.x * T, -l.z * T);
    }
    if (s.drop && s.phase !== "won") {
      const m = s.drop, bob = m.rest ? Math.sin(clock * 4) * 0.08 : 0;
      ctx.save();
      ctx.shadowColor = RED;
      ctx.shadowBlur = 12 + Math.sin(clock * 5) * 6;
      ctx.fillStyle = RED;
      ctx.font = "800 44px system-ui, -apple-system, 'Segoe UI', sans-serif";
      ctx.fillText(m.ch, m.x * T, -(m.z + 0.7 + bob) * T);
      ctx.restore();
    }
  }

  /* ---------- the world ---------- */

  function hill(left, right, parallax, base, amp, color) {
    const off = left * T * (1 - parallax);
    ctx.fillStyle = color;
    ctx.beginPath();
    const x0 = left * T, x1 = right * T;
    ctx.moveTo(x0, 40);
    for (let x = x0; x <= x1 + 24; x += 24) {
      const u = (x - off) / 180;
      ctx.lineTo(x, -(base + amp * (Math.sin(u) * 0.6 + Math.sin(u * 0.43 + 1) * 0.9)) * T);
    }
    ctx.lineTo(x1 + 24, 40);
    ctx.fill();
  }

  function drawBackdrop(left, right, bottom, top) {
    ctx.fillStyle = PAPER;
    ctx.fillRect(left * T, -top * T, (right - left) * T, (top - bottom) * T);
    hill(left, right, 0.3, 4.2, 1.5, "#eeeadd");
    hill(left, right, 0.55, 2.4, 1.1, "#e8e4d5");

    /* one square is one centimetre, every fifth line is stronger */
    ctx.lineWidth = 1 / view.zoom;
    for (let x = Math.floor(left); x <= right; x++) {
      ctx.strokeStyle = x % 5 === 0 ? "rgba(42,41,57,0.13)" : "rgba(42,41,57,0.05)";
      ctx.beginPath();
      ctx.moveTo(x * T, -top * T);
      ctx.lineTo(x * T, -bottom * T);
      ctx.stroke();
    }
    for (let z = Math.floor(bottom); z <= top; z++) {
      ctx.strokeStyle = z % 5 === 0 ? "rgba(42,41,57,0.13)" : "rgba(42,41,57,0.05)";
      ctx.beginPath();
      ctx.moveTo(left * T, -z * T);
      ctx.lineTo(right * T, -z * T);
      ctx.stroke();
    }

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(heat, -4 * T, -Z_MAX * T, (LEVEL_W + 8) * T, Z_MAX * T);

    ctx.strokeStyle = "rgba(31,116,196,0.62)";
    ctx.lineWidth = 1.3;
    ctx.setLineDash([7, 9]);
    ctx.lineDashOffset = reduced ? 0 : -clock * 34;
    for (const l of lines) {
      if (l.x1 < left || l.x0 > right) continue;
      ctx.stroke(l.path);
    }
    ctx.setLineDash([]);
  }

  function drawSolid(q) {
    const x = q.x0 * T, y = -q.z1 * T, w = (q.x1 - q.x0) * T, h = (q.z1 - q.z0) * T;
    if (q.kind === "post") {
      ctx.fillStyle = INK;
      ctx.fillRect(x, y, w, h);
      return;
    }
    ctx.fillStyle = INK;
    ctx.fillRect(x, y, w, h);
    ctx.fillStyle = "#454459";
    ctx.fillRect(x, y, w, 5);
    if (q.kind === "crate") {
      ctx.strokeStyle = "#454459";
      ctx.lineWidth = 2;
      ctx.strokeRect(x + 6, y + 10, w - 12, h - 16);
    }
    /* solder pads along the ground */
    ctx.fillStyle = GOLD;
    for (let i = Math.ceil(q.x0 / 2) * 2 + 1; i < q.x1; i += 2) {
      if (h > T * 2) {
        ctx.beginPath();
        ctx.arc(i * T, y + 18, 3, 0, 7);
        ctx.fill();
      }
    }
  }

  function poleGradient(x, y0, y1, up) {
    const g = ctx.createLinearGradient(x, y0, x, y1);
    const a = up ? NORTH : SOUTH, b = up ? SOUTH : NORTH;
    g.addColorStop(0, a);
    g.addColorStop(0.42, a);
    g.addColorStop(0.5, MIDDLE);
    g.addColorStop(0.58, b);
    g.addColorStop(1, b);
    return g;
  }

  function magnetText(m, x, y) {
    ctx.fillStyle = "#fff";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = "700 13px ui-monospace, Menlo, Consolas, monospace";
    ctx.save();
    ctx.shadowColor = "rgba(0,0,0,0.55)";
    ctx.shadowBlur = 3;
    ctx.fillText(m.name, x, y);
    ctx.restore();
  }

  function drawMagnet(m) {
    const up = m.J[2] > 0;
    if (m.kind === "sphere") {
      const r = (m.dim / 2) * T, x = m.x * T, y = -m.z * T;
      ctx.fillStyle = poleGradient(x, y - r, y + r, up);
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 7);
      ctx.fill();
      ctx.strokeStyle = "rgba(42,41,57,0.5)";
      ctx.lineWidth = 2;
      ctx.stroke();
      magnetText(m, x, y);
      return;
    }
    const [w, h] = m.kind === "cuboid" ? [m.dim[0], m.dim[2]] : m.dim;
    const x = (m.x - w / 2) * T, y = -(m.z + h / 2) * T;
    ctx.fillStyle = poleGradient(x, y, y + h * T, up);
    ctx.fillRect(x, y, w * T, h * T);
    ctx.strokeStyle = "rgba(42,41,57,0.5)";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w * T, h * T);
    /* buried cylinders show their name where the player can see it */
    const label = m.kind === "cylinder" ? y + 22 : y + (h * T) / 2;
    magnetText(m, m.x * T, label);
  }

  function drawSensor(c) {
    const x = c.x * T, y = -(c.z + Math.sin(clock * 3 + c.x) * 0.07) * T;
    ctx.fillStyle = GOLD;
    for (const dx of [-6, 0, 6]) {
      ctx.fillRect(x + dx - 1.5, y - 12, 3, 24);
    }
    ctx.fillStyle = INK;
    ctx.fillRect(x - 10, y - 7, 20, 14);
    ctx.fillStyle = GOLD;
    ctx.beginPath();
    ctx.arc(x - 6, y - 3, 1.6, 0, 7);
    ctx.fill();
  }

  function drawNut(n) {
    const x = n.x * T, y = -n.z * T;
    if (n.dead) {
      if (n.dead < 0.4) return;
      ctx.fillStyle = "#7a7a88";
      ctx.fillRect(x - 15, y - 5, 30, 5);
      return;
    }
    const r = 15, wob = Math.sin(clock * 9 + n.x) * 2.5;
    ctx.fillStyle = INK;
    ctx.fillRect(x - 11 + wob, y - 4, 7, 4);
    ctx.fillRect(x + 4 - wob, y - 4, 7, 4);
    ctx.fillStyle = "#8a8b98";
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const a = (i * Math.PI) / 3;
      ctx.lineTo(x + r * Math.cos(a), y - 18 + r * Math.sin(a));
    }
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = PAPER;
    ctx.beginPath();
    ctx.arc(x, y - 18, 6.5, 0, 7);
    ctx.fill();
    ctx.fillStyle = INK;
    ctx.beginPath();
    ctx.arc(x + n.dir * 2.5, y - 18, 3, 0, 7);
    ctx.fill();
  }

  function drawCheck(cp) {
    const up = s.check.x >= cp.x, x = cp.x * T, y = -cp.z * T;
    ctx.fillStyle = INK;
    ctx.fillRect(x - 1.5, y - 72, 3, 72);
    ctx.fillStyle = up ? RED : "#b9b6a8";
    ctx.beginPath();
    ctx.moveTo(x + 1.5, y - 72);
    ctx.lineTo(x + 24, y - 63);
    ctx.lineTo(x + 1.5, y - 54);
    ctx.fill();
  }

  /* ---------- on top of the view ---------- */

  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, r);
  }

  const FLIP_NAME = touchUI ? "Flip" : "F";
  const HINTS = [
    [37, () => `Thief! Get our letters back. ${touchUI ? "Use the buttons below." : "← → to walk, Space to jump."}`],
    [42, () => "A Cuboid. You are a magnet too, and that pull is its real field."],
    [50, () => `Stuck on it? Press ${FLIP_NAME} to flip your magnet: pull becomes push.`],
    [59, () => "Jump under the long Cuboid. The field catches you."],
    [81, () => `Walk across upside down, and press ${FLIP_NAME} over the far ground to let go.`],
    [91, () => `Cylinders. Press ${FLIP_NAME} to push, then hover on the field.`],
    [121, () => "Pushing, you can jump straight off the field. Hop across."],
    [137, () => `A Sphere. Pull: stick to it, walk to the top. Then press ${FLIP_NAME}.`],
    [Infinity, () => (s.drop ? "It dropped the M! Grab it." : "There's the magpie.")],
  ];

  function drawHint(dt) {
    hintText = s.phase === "play" ? HINTS.find((h) => s.p.x < h[0])[1]() : "";
    if (hintText !== hintShown) {
      hintFade -= dt * 6;
      if (hintFade <= 0) {
        hintFade = 0;
        hintShown = hintText;
      }
    } else {
      hintFade = Math.min(1, hintFade + dt * 5);
    }
    if (!hintShown || hintFade <= 0) return;
    ctx.save();
    ctx.globalAlpha = hintFade;
    ctx.font = "600 17px system-ui, -apple-system, 'Segoe UI', sans-serif";
    const w = ctx.measureText(hintShown).width + 36;
    roundRect(VW / 2 - w / 2, 58, w, 36, 18);
    ctx.fillStyle = "rgba(42,41,57,0.92)";
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(hintShown, VW / 2, 76.5);
    ctx.restore();
  }

  function slot(i) {
    return { x: 22 + i * 21, y: 27 };
  }

  function drawHUD() {
    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    ctx.font = "800 22px system-ui, -apple-system, 'Segoe UI', sans-serif";
    [..."MAGPYLIB"].forEach((ch, i) => {
      const lit = i === 0 && s.phase === "won" && winT > 0.8;
      ctx.fillStyle = lit ? RED : "rgba(42,41,57,0.2)";
      ctx.fillText(ch, slot(i).x, slot(i).y);
    });

    ctx.textAlign = "left";
    ctx.font = "700 15px ui-monospace, Menlo, Consolas, monospace";
    ctx.save();
    ctx.translate(205, 27);
    ctx.fillStyle = GOLD;
    for (const dx of [-5, 0, 5]) ctx.fillRect(dx - 1, -10, 2, 20);
    ctx.fillStyle = INK;
    ctx.fillRect(-8, -6, 16, 12);
    ctx.restore();
    ctx.fillStyle = INK;
    ctx.fillText(`${s.got}`, 220, 28);

    /* the reading: same |B| the force is the gradient of */
    const b = Sim.absB(s.p.x, s.p.z + H / 2);
    const px = VW - 262, py = 12;
    roundRect(px, py, 180, 30, 8);
    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.fill();
    ctx.fillStyle = INK;
    ctx.fillText(`|B| ${b.toFixed(3).padStart(5)} T`, px + 10, py + 15.5);
    const bar = Math.max(0, Math.min(1, (Math.log10(b + 1e-9) + 2.6) / 2.4));
    ctx.fillStyle = "rgba(42,41,57,0.12)";
    ctx.fillRect(px + 124, py + 11, 46, 8);
    ctx.fillStyle = BLUE;
    ctx.fillRect(px + 124, py + 11, 46 * bar, 8);

    const pull = s.p.pol > 0;
    roundRect(VW - 74, py, 62, 30, 8);
    ctx.fillStyle = pull ? RED : PUSH;
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.textAlign = "center";
    ctx.fillText(pull ? "PULL" : "PUSH", VW - 43, py + 15.5);

    /* scale bar */
    const sb = T * view.zoom;
    roundRect(10, VH - 36, sb + 58, 26, 7);
    ctx.fillStyle = "rgba(255,255,255,0.88)";
    ctx.fill();
    ctx.fillStyle = INK;
    ctx.fillRect(18, VH - 24, sb, 2);
    ctx.fillRect(18, VH - 28, 2, 10);
    ctx.fillRect(18 + sb - 2, VH - 28, 2, 10);
    ctx.textAlign = "left";
    ctx.font = "600 12px ui-monospace, Menlo, Consolas, monospace";
    ctx.fillText("1 cm", 18 + sb + 8, VH - 22);
  }

  function drawTitle() {
    const a = s.phase === "title" ? 1 : Math.max(0, 1 - s.t * 2.5);
    if (a <= 0) return;
    ctx.save();
    ctx.globalAlpha = a;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = "900 76px system-ui, -apple-system, 'Segoe UI', sans-serif";
    ctx.fillStyle = INK;
    const left = "Super ", right = "Magrio";
    const wl = ctx.measureText(left).width, wr = ctx.measureText(right).width;
    const x0 = VW / 2 - (wl + wr) / 2;
    ctx.textAlign = "left";
    ctx.fillText(left, x0, 76);
    ctx.fillStyle = RED;
    ctx.fillText(right, x0 + wl, 76);
    ctx.textAlign = "center";
    ctx.fillStyle = INK;
    ctx.font = "500 21px system-ui, -apple-system, 'Segoe UI', sans-serif";
    ctx.fillText("The magpie took our letters. Get them back.", VW / 2, 130);
    if (s.phase === "title" && Math.floor(clock * 1.6) % 2 === 0) {
      ctx.font = "700 22px system-ui, -apple-system, 'Segoe UI', sans-serif";
      ctx.fillStyle = "#fff";
      ctx.fillText(touchUI ? "Tap to start" : "Press Space to start", VW / 2, VH - 42);
    }
    ctx.restore();
  }

  function drawWinLetter() {
    if (s.phase !== "won" || !s.drop) return;
    const t = Math.min(1, winT / 0.8), e = t * t * (3 - 2 * t);
    const from = toScreen(s.drop.x, s.drop.z + 0.7), to = slot(0);
    ctx.fillStyle = RED;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = `800 ${44 - 22 * e}px system-ui, -apple-system, 'Segoe UI', sans-serif`;
    if (t < 1) ctx.fillText("M", from.x + (to.x - from.x) * e, from.y + (to.y - from.y) * e - Math.sin(e * Math.PI) * 60);
  }

  function drawWon() {
    if (s.phase !== "won" || winT < WON_WAIT) return;
    ctx.save();
    ctx.globalAlpha = Math.min(1, (winT - WON_WAIT) * 4);
    roundRect(VW / 2 - 250, VH / 2 - 86, 500, 172, 18);
    ctx.fillStyle = "rgba(42,41,57,0.94)";
    ctx.fill();
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#fff";
    ctx.font = "800 36px system-ui, -apple-system, 'Segoe UI', sans-serif";
    ctx.fillText("You got the M back!", VW / 2, VH / 2 - 42);
    ctx.font = "500 19px system-ui, -apple-system, 'Segoe UI', sans-serif";
    ctx.fillText("The magpie still has the other seven.", VW / 2, VH / 2 + 2);
    ctx.font = "700 18px system-ui, -apple-system, 'Segoe UI', sans-serif";
    ctx.fillStyle = GOLD;
    ctx.fillText(touchUI ? "Tap to play again" : "Press Space to play again", VW / 2, VH / 2 + 46);
    ctx.restore();
  }

  function burst(x, z, color, n = 8) {
    if (reduced) return;
    for (let i = 0; i < n; i++) {
      const a = Math.random() * 6.3, v = 2 + Math.random() * 5;
      particles.push({ x, z, vx: Math.cos(a) * v, vz: Math.sin(a) * v, t: 0.5, color });
    }
  }

  function drawParticles(dt) {
    particles = particles.filter((q) => (q.t -= dt) > 0);
    for (const q of particles) {
      q.x += q.vx * dt;
      q.z += q.vz * dt;
      q.vz -= 14 * dt;
      ctx.fillStyle = q.color;
      ctx.globalAlpha = Math.min(1, q.t * 3);
      ctx.fillRect(q.x * T - 2, -q.z * T - 2, 4, 4);
    }
    ctx.globalAlpha = 1;
  }

  /* ---------- camera ---------- */

  function viewBox() {
    const w = VW / (T * view.zoom), h = VH / (T * view.zoom);
    return { left: view.x - w / 2, right: view.x + w / 2, bottom: view.z, top: view.z + h };
  }

  function toScreen(x, z) {
    const b = viewBox();
    return { x: (x - b.left) * T * view.zoom, y: VH - (z - b.bottom) * T * view.zoom };
  }

  function updateCamera(dt) {
    const p = s.p, title = s.phase === "title";
    const into = title ? 0 : Math.min(1, s.t / 1.5), e = into * into * (3 - 2 * into);
    view.zoom = 2.1 + (1 - 2.1) * e;
    const halfW = VW / (T * view.zoom) / 2;

    let tx, tz;
    if (s.phase === "intro" || title) {
      tx = 13 + (Math.max(p.x + 3, 13.34) - 13) * e;
      tz = 0.6 * (1 - e);
      view.x = tx;
      view.z = tz;
    } else {
      tx = p.x + p.face * 2.5 + p.vx * 0.12;
      tz = Math.max(0, p.z + H + 2 - VH / T);
      const k = 1 - Math.exp(-dt * 5);
      view.x += (tx - view.x) * k;
      view.z += (tz - view.z) * (1 - Math.exp(-dt * 6));
    }
    view.x = Math.max(-6 + halfW, Math.min(LEVEL_W - halfW, view.x));
  }

  /* ---------- one frame ---------- */

  function draw(dt) {
    const scale = canvas.width / VW;
    const b = viewBox();
    ctx.setTransform(scale * view.zoom, 0, 0, scale * view.zoom, 0, 0);
    ctx.translate(-b.left * T, b.top * T);

    drawBackdrop(b.left, b.right, b.bottom, b.top);
    for (const cp of CHECKPOINTS) drawCheck(cp);
    for (const q of SOLIDS) {
      if (!q.magnet && q.x1 > b.left && q.x0 < b.right) drawSolid(q);
    }
    for (const m of MAGNETS) {
      const half = m.kind === "cuboid" ? m.dim[0] / 2 : 4;
      if (m.x + half > b.left && m.x - half < b.right) drawMagnet(m);
    }
    for (const c of s.sensors) if (!c.got && c.x > b.left - 1 && c.x < b.right + 1) drawSensor(c);
    for (const n of s.nuts) if (n.x > b.left - 1 && n.x < b.right + 1) drawNut(n);
    drawLetters();
    drawBird();
    drawHero();
    drawParticles(dt);

    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    drawHint(dt);
    if (s.phase !== "title") drawHUD();
    drawTitle();
    drawWinLetter();
    drawWon();
  }

  /* ---------- input ---------- */

  const KEYS = {
    ArrowLeft: "left", KeyA: "left", ArrowRight: "right", KeyD: "right",
    Space: "jump", ArrowUp: "jump", KeyW: "jump", KeyF: "flip", KeyX: "flip",
  };

  function playAgain() {
    if (s.phase !== "won" || winT < WON_WAIT) return;
    s = Sim.create();
    winT = 0;
    Object.assign(view, { x: 13, z: 0.6, zoom: 2.1 });
  }

  function wake() {
    if (!audio) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (AC) audio = new AC();
    }
    if (audio && audio.state === "suspended") audio.resume();
    if (s.phase === "title") Sim.start(s);
  }

  addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (document.activeElement && document.activeElement.tagName === "BUTTON" && e.code === "Space") return;
    const k = KEYS[e.code];
    if (k) {
      input[k] = true;
      e.preventDefault();
      if (e.repeat) return;
      tapped[k] = true;
      if (k === "jump") playAgain();
      if (k === "jump" || s.phase !== "title") wake();
    }
  });
  addEventListener("keyup", (e) => {
    const k = KEYS[e.code];
    if (k) input[k] = false;
  });
  addEventListener("blur", () => {
    input.left = input.right = input.jump = input.flip = false;
  });
  canvas.addEventListener("pointerdown", () => {
    playAgain();
    wake();
  });

  for (const btn of document.querySelectorAll("[data-key]")) {
    const k = btn.dataset.key;
    const on = (e) => {
      e.preventDefault();
      input[k] = true;
      tapped[k] = true;
      if (k === "jump") playAgain();
      wake();
      btn.classList.add("on");
    };
    const off = (e) => { e.preventDefault(); input[k] = false; btn.classList.remove("on"); };
    btn.addEventListener("pointerdown", on);
    btn.addEventListener("pointerup", off);
    btn.addEventListener("pointercancel", off);
    btn.addEventListener("pointerleave", off);
  }

  const soundBtn = document.getElementById("sound");
  soundBtn.addEventListener("click", () => {
    muted = !muted;
    soundBtn.textContent = muted ? "Sound: off" : "Sound: on";
    soundBtn.blur();
  });

  function resize() {
    const ratio = Math.min(devicePixelRatio || 1, 2);
    const box = canvas.getBoundingClientRect();
    canvas.width = Math.round(box.width * ratio);
    canvas.height = Math.round(box.height * ratio);
  }
  addEventListener("resize", resize);

  /* ---------- the loop ---------- */

  let last = 0, acc = 0;

  function consumeEvents() {
    for (const ev of s.events) {
      if (SOUND[ev]) SOUND[ev]();
      const p = s.p, cx = p.x, cz = p.z + H / 2;
      if (ev === "sensor") burst(cx, cz, GOLD, 7);
      else if (ev === "flip") burst(cx, cz, p.pol > 0 ? RED : PUSH, 10);
      else if (ev === "stomp") burst(cx, p.z, "#8a8b98", 8);
      else if (ev === "win") burst(cx, cz, RED, 24);
    }
    s.events.length = 0;
  }

  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000 || 0);
    last = now;
    clock += dt;
    acc += dt;
    while (acc >= Sim.STEP) {
      Sim.step(s, {
        left: input.left,
        right: input.right,
        jump: input.jump || tapped.jump,
        flip: input.flip || tapped.flip,
      });
      tapped.jump = tapped.flip = false;
      acc -= Sim.STEP;
    }
    consumeEvents();
    if (s.phase === "won") winT += dt;
    updateCamera(dt);
    draw(dt);
    requestAnimationFrame(frame);
  }

  resize();
  const bakeStart = performance.now();
  bakeHeat();
  bakeLines();
  const bakeMs = performance.now() - bakeStart;
  requestAnimationFrame(frame);

  /* for the screenshot check, no effect on play */
  globalThis.Magrio = {
    get state() { return s; },
    input,
    view,
    bakeMs,
  };
})();
