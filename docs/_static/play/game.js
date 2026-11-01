/*
 * Super Magrio: drawing, input and sound on top of sim.js.
 *
 * Canvas pixels are 36 to the tile at zoom 1, in a view 540 high and as wide as the
 * screen allows. The background glow and the field lines are computed once from the
 * same formulas the hero feels, so what is drawn is what pushes. Colours follow
 * Magpylib's: north is red, south is green, for magnets, coils and the hero alike.
 */
(() => {
  "use strict";

  const Sim = globalThis.MagrioSim, Field = globalThis.MagField;
  const { H, BIG, GROUND, LEVEL_W, MAGNETS, SOLIDS, CHECKPOINTS, BORE, RIM } = Sim;

  const T = 36, VH = 540, Z_MAX = 26;
  let VW = 960;
  const RED = "#aa0000", INK = "#2a2939", PAPER = "#f6f3ea", BLUE = "#2e91e5";
  const NORTH = "#e71111", MIDDLE = "#dddddd", SOUTH = "#00b050", GOLD = "#d9a441";
  const COPPER = "#c8763a", COPPER_LIT = "#eaa873";
  /* Field lines as Magpylib Studio draws them: fine solid lines, stronger where the field
     is, gone where it fades, with a short pulse running the way B points. Quieter here,
     a hint of where the magnets reach and no more. */
  const LINE_RGB = "59,79,160", LINE_END = 0.03;
  const LINE_BANDS = [
    { below: 0.06, alpha: 0.16, width: 1 },
    { below: 0.14, alpha: 0.3, width: 1.2 },
    { below: Infinity, alpha: 0.5, width: 1.5 },
  ];
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
  const input = { left: false, right: false, jump: false, flip: false, fire: false };
  /* a press stays noted until a step has read it, however short the tap and slow the frame */
  const tapped = { jump: false, flip: false, fire: false };
  const view = { x: 13, z: 0.6, zoom: 2.1 };
  /* the title frames the logo and the wordmark, about 10.5 tiles across */
  const titleZoom = () => Math.min(2.1, VW / (10.5 * T));
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
        const b = Sim.absB(i / HEAT_RES - 4, z, Sim.SPHERE);
        const t = Math.max(0, Math.min(1, (Math.log10(b + 1e-9) + 1.7) / 1.7));
        img.data[k] = 46;
        img.data[k + 1] = 120;
        img.data[k + 2] = 220;
        img.data[k + 3] = 62 * t * t;
      }
    }
    g.putImageData(img, 0, 0);
  }

  const out = [0, 0, 0];
  let alone = null; /* bake lines for this magnet by itself: the sphere's turn with it */
  function B2(x, z) {
    out[0] = out[1] = out[2] = 0;
    for (const m of alone ? [alone] : MAGNETS) {
      if (Math.abs(x - m.x) < 40 + m.half) Field.addB(m, x, 0, z, out);
    }
    return out;
  }

  function insideMagnet(x, z) {
    for (const m of MAGNETS) {
      if (m.kind === "coil") continue; /* lines run through the bore */
      if (m.kind === "sphere") {
        if (Math.hypot(x - m.x, z - m.z) < m.dim / 2) return true;
      } else {
        const [w, h] = m.kind === "cuboid" ? [m.dim[0], m.dim[2]] : m.dim;
        if (Math.abs(x - m.x) < w / 2 && Math.abs(z - m.z) < h / 2) return true;
      }
    }
    return false;
  }

  /* follows B, or against it for sign -1, until it lands on something, closes or fades */
  function trace(x, z, sign) {
    const h = 0.2 * sign, pts = [[x, z]], x0 = x, z0 = z;
    for (let i = 0; i < 900; i++) {
      let b = B2(x, z);
      let n = Math.hypot(b[0], b[2]) || 1;
      const mx = x + (0.5 * h * b[0]) / n, mz = z + (0.5 * h * b[2]) / n;
      b = B2(mx, mz);
      n = Math.hypot(b[0], b[2]) || 1;
      x += (h * b[0]) / n;
      z += (h * b[2]) / n;
      pts.push([x, z]);
      if ((!alone && (z < -1 || z > Z_MAX)) || insideMagnet(x, z)) break;
      if (i > 30 && Math.hypot(x - x0, z - z0) < 0.3) break;
      const here = B2(x, z);
      if (Math.hypot(here[0], here[2]) < LINE_END) break;
    }
    return sign > 0 ? pts : pts.reverse();
  }

  /* [x, z, sign] to start a line from: north faces outwards, and any south face in view */
  function seeds(m) {
    const list = [];
    if (m.kind === "coil") {
      const top = m.z + m.h / 2, sign = m.I > 0 ? 1 : -1;
      for (const dx of [-0.8, 0, 0.8]) list.push([m.x + dx, top, sign]);
      return list;
    }
    if (m.kind === "sphere") {
      for (const deg of [-56, -30, 0, 30, 56]) {
        const a = (deg * Math.PI) / 180, r = m.dim / 2 + 0.03;
        list.push([m.x + r * Math.sin(a), m.z + r * Math.cos(a), 1]);
      }
      return list;
    }
    const [w, h] = m.kind === "cuboid" ? [m.dim[0], m.dim[2]] : m.dim;
    const n = Math.min(12, Math.max(3, Math.round(w / 2.4)));
    for (let i = 0; i < n; i++) {
      const x = m.x + (w * (i + 0.5)) / n - w / 2, sign = m.J[2] > 0 ? 1 : -1;
      list.push([x, m.z + h / 2 + 0.03, sign]);
      if (m.z - h / 2 > GROUND) list.push([x, m.z - h / 2 - 0.03, -sign]);
    }
    return list;
  }

  const lines = [];
  function bakeLines() {
    for (const m of MAGNETS) {
      const bands = LINE_BANDS.map(() => new Path2D());
      const whole = new Path2D();
      let x0 = Infinity, x1 = -Infinity;
      alone = m === Sim.SPHERE ? m : null;
      for (const [sx, sz, sign] of seeds(m)) {
        trace(sx, sz, sign).forEach(([x, z], i, pts) => {
          if (i) whole.lineTo(x * T, -z * T);
          else whole.moveTo(x * T, -z * T);
          x0 = Math.min(x0, x);
          x1 = Math.max(x1, x);
          if (!i) return;
          const b = B2(x, z), mag = Math.hypot(b[0], b[2]);
          const path = bands[LINE_BANDS.findIndex((band) => mag < band.below)];
          path.moveTo(pts[i - 1][0] * T, -pts[i - 1][1] * T);
          path.lineTo(x * T, -z * T);
        });
      }
      lines.push(alone ? { bands, whole, x0: m.x - 12, x1: m.x + 12, turns: m } : { bands, whole, x0, x1 });
    }
    alone = null;
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
    stomp: () => [880, 660, 440, 330].forEach((f, i) => setTimeout(() => blip(f, f * 0.8, 0.08, "triangle", 0.06), i * 55)),
    grab: () => blip(150, 90, 0.3, "sawtooth", 0.08),
    blast: () => blip(260, 1500, 0.24, "sawtooth", 0.05),
    shoot: () => blip(1400, 500, 0.12, "square", 0.04),
    hurt: () => blip(420, 90, 0.34, "triangle", 0.09),
    squawk: () => { blip(1250, 700, 0.09, "sawtooth", 0.025); setTimeout(() => blip(1100, 620, 0.12, "sawtooth", 0.025), 110); },
    pluck: () => blip(880, 1400, 0.05, "triangle", 0.04),
    check: () => blip(520, 780, 0.2, "triangle", 0.06),
    clamp: () => blip(240, 120, 0.07, "square", 0.06),
    clink: () => blip(1700, 1500, 0.08, "triangle", 0.05),
    bump: () => blip(300, 520, 0.12, "square", 0.05),
    grow: () => [392, 523, 659].forEach((f, i) => setTimeout(() => blip(f, f * 1.05, 0.09, "square", 0.05), i * 70)),
    shrink: () => blip(660, 180, 0.3, "square", 0.05),
    warp: () => blip(140, 900, 0.3, "sawtooth", 0.04),
    win: () => [523, 659, 784, 1047].forEach((f, i) => setTimeout(() => blip(f, f, 0.2, "triangle", 0.08), i * 110)),
  };

  /* ---------- the hero: the logo's magnet, on its pole tips ---------- */

  function rect(x0, y0, x1, y1) {
    ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
  }

  /* The force shows on the hero, eased so that it never snaps. Its eyes watch the magnet.
     Drawn in, it stretches towards it wide-eyed, and flattens on arriving. Pushed away,
     it squints and braces against the push. */
  let grown = 1;
  const cue = { s: 0, ang: 0, lookX: 7, lookY: 0, eye: 1, wide: 1 };

  function updateCue(p, dt) {
    const a = Math.hypot(p.a.x, p.a.z), g = s.phase === "play" ? a / 38 : 0;
    const m = Math.min(1, Math.max(0, (g - 0.12) / 1.6));
    const contact = p.grounded || p.hang;
    let stretch = 0, lookX = p.face * 7, lookY = 0, eye = 1, wide = 1;
    if (m > 0) {
      const ux = p.a.x / a, uy = -p.a.z / a; /* towards the magnet if drawn in, away if pushed */
      const sign = p.attract ? 1 : -1;
      stretch = p.attract ? (contact ? -0.16 : 0.22) * m : (contact ? 0.08 : -0.15) * m;
      eye = p.attract ? 1 : 1 - 0.5 * m;
      wide = p.attract ? 1 + 0.18 * m : 1;
      const c = Math.cos(p.tilt), n = Math.sin(p.tilt), w = Math.min(1, m * 2.5);
      lookX += ((sign * (ux * c + uy * n) * 8) - lookX) * w;
      lookY += (sign * (-ux * n + uy * c) * 7 - lookY) * w;
      /* the short way round, so the axis never spins through a turn */
      let da = Math.atan2(uy, ux) - cue.ang;
      da = Math.atan2(Math.sin(2 * da), Math.cos(2 * da)) / 2;
      cue.ang += da * (1 - Math.exp(-dt * 14));
    }
    const k = 1 - Math.exp(-dt * 12);
    cue.s += (stretch - cue.s) * k;
    cue.lookX += (lookX - cue.lookX) * k;
    cue.lookY += (lookY - cue.lookY) * k;
    cue.eye += (eye - cue.eye) * k;
    cue.wide += (wide - cue.wide) * k;
  }

  function drawMagnetBody(top, feet, liftL, liftR, eyes) {
    const { arch, outer, inner, cx } = LOGO;
    ctx.fillStyle = top;
    ctx.beginPath();
    ctx.arc(cx, arch, outer, Math.PI, 0);
    ctx.arc(cx, arch, inner, 0, Math.PI, true);
    ctx.closePath();
    ctx.fill();
    /* the legs shorten as the feet lift, the white gap above each foot stays. Clamping, they
       bend in from the arch until the feet nearly meet. */
    const reach = s.phase === "title" ? 0 : s.p.pinch * 54;
    const inward = (y) => reach * Math.max(0, (y - arch) / (340 - arch));
    const leg = (side, lift) => {
      const end = 264 - lift, a = cx + side * outer, b = cx + side * inner, d = -side * inward(end);
      ctx.beginPath();
      ctx.moveTo(a, arch - 1);
      ctx.lineTo(b, arch - 1);
      ctx.lineTo(b + d, end);
      ctx.lineTo(a + d, end);
      ctx.closePath();
      ctx.fill();
    };
    leg(-1, liftL);
    leg(1, liftR);
    if (s.p.fire) {
      /* earned the current loop: its copper is wound round the legs */
      ctx.fillStyle = COPPER;
      for (const y of [186, 208, 230]) {
        const d = inward(y);
        rect(cx - outer - 4 + d, y, cx - inner + 4 + d, y + 13);
        rect(cx + inner - 4 - d, y, cx + outer + 4 - d, y + 13);
      }
    }
    ctx.fillStyle = feet;
    const footL = inward(340 - liftL), footR = inward(340 - liftR);
    rect(cx - outer + footL, 284 - liftL, cx - inner + footL, 340 - liftL);
    rect(cx + inner - footR, 284 - liftR, cx + outer - footR, 340 - liftR);

    if (eyes > 0) {
      const blink = (clock % 3.7) < 0.1 ? 0.15 : 1, open = eyes * blink * cue.eye;
      for (const ex of [cx - 30, cx + 30]) {
        ctx.fillStyle = "#fff";
        ctx.beginPath();
        ctx.ellipse(ex, 86, 19 * cue.wide, 19 * cue.wide * open, 0, 0, 7);
        ctx.fill();
        ctx.fillStyle = INK;
        ctx.beginPath();
        ctx.ellipse(ex + cue.lookX, 87 + cue.lookY, 10, 10 * open, 0, 0, 7);
        ctx.fill();
      }
    }
  }

  function drawHero(dt) {
    const p = s.p;
    updateCue(p, dt);
    if (p.safe && !p.grab && Math.floor(clock * 14) % 2) return;
    grown += ((p.big ? BIG : 1) - grown) * (1 - Math.exp(-dt * 10));
    const k = LOGO.s * T * grown;
    ctx.save();
    /* leaning on flat ground, the lower foot rests on it rather than digging in */
    const lift = p.onSphere ? 0 : Math.max(0, -p.pivot) * (p.w / 2) * Math.abs(Math.sin(p.tilt));
    /* held by an enemy it shakes */
    const shake = p.grab ? Math.sin(clock * 70) * 2.5 : 0;
    ctx.translate(p.x * T + shake, -(p.z + lift + (p.h * (1 + p.pivot)) / 2) * T);
    ctx.rotate(cue.ang);
    ctx.scale(1 + cue.s, 1 - cue.s * 0.55);
    ctx.rotate(p.tilt - cue.ang);
    ctx.translate(0, ((p.pivot * p.h) / 2) * T);
    const flip = p.flipT > 0 ? Math.abs(Math.cos((Math.PI * p.flipT) / 0.28)) : 1;
    const squash = p.land / 0.16;
    ctx.scale(k * Math.max(0.12, flip) * (1 + 0.12 * squash), k * (1 - 0.14 * squash));
    ctx.translate(-LOGO.cx, -LOGO.cz);
    const moving = p.grounded || p.hang, step = Math.sin(p.walk);
    let liftL = 0, liftR = 0;
    if (s.phase === "play" || s.phase === "won") {
      if (!moving) liftL = liftR = 24;
      else if (Math.abs(p.vx) > 0.8) {
        liftL = Math.max(0, step) * 34;
        liftR = Math.max(0, -step) * 34;
      }
    }
    const eyes = s.phase === "title" ? 0 : Math.min(1, s.t * 3);
    /* on the title it is the logo, all one red. Awake it has poles: a real magnet does */
    if (s.phase === "title") drawMagnetBody(RED, RED, 0, 0, 0);
    else drawMagnetBody(p.pol > 0 ? NORTH : SOUTH, p.pol > 0 ? SOUTH : NORTH, liftL, liftR, eyes);
    if (s.bird.mode === "logo" && birdStill.complete) {
      ctx.drawImage(birdStill, BIRD_BOX.x, BIRD_BOX.y);
    }
    ctx.restore();

  }

  /* ---------- the magpie ---------- */

  function drawBird() {
    const b = s.bird;
    if (b.mode === "logo" || b.mode === "gone") return;
    drawBag(b);
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
      const f = Math.floor(b.flap * 13) % FRAME.n;
      if (birdFly.complete) {
        ctx.drawImage(birdFly, f * FRAME.w, 0, FRAME.w, FRAME.h, -FRAME.w / 2, -FRAME.h / 2, FRAME.w, FRAME.h);
      }
    }
    ctx.restore();
  }

  function drawBag(b) {
    const [x, z] = Sim.bagAt(b);
    const perched = b.mode === "perched";
    const r = (0.3 + 0.03 * b.loot) * T * (1 + (b.bump > 0 ? 0.25 * (b.bump / 0.14) : 0)) * Math.min(1, b.flap * 4);
    const cx = x * T, cy = -z * T;
    if (!perched) {
      /* the cord up to the claws */
      ctx.strokeStyle = INK;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(b.x * T, -(b.z - 0.5) * T);
      ctx.lineTo(cx, cy - r * 0.9);
      ctx.stroke();
    }
    ctx.fillStyle = INK;
    ctx.beginPath();
    ctx.moveTo(cx - r * 0.3, cy - r * 0.95);
    ctx.bezierCurveTo(cx - r * 1.3, cy - r * 0.5, cx - r * 1.15, cy + r, cx, cy + r);
    ctx.bezierCurveTo(cx + r * 1.15, cy + r, cx + r * 1.3, cy - r * 0.5, cx + r * 0.3, cy - r * 0.95);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = GOLD;
    ctx.fillRect(cx - r * 0.42, cy - r * 0.98, r * 0.84, r * 0.2);
  }

  function drawLetters() {
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = INK;
    for (const l of s.letters) {
      if (l.t === 1) continue;
      ctx.font = `800 ${34 * l.size}px system-ui, -apple-system, 'Segoe UI', sans-serif`;
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
    /* the sphere turns, so its glow is round rather than baked */
    const ball = Sim.SPHERE, bx = ball.x * T, by = -ball.z * T, br = (ball.dim / 2) * T;
    const glow = ctx.createRadialGradient(bx, by, br, bx, by, br * 2.1);
    glow.addColorStop(0, "rgba(46,120,220,0.2)");
    glow.addColorStop(1, "rgba(46,120,220,0)");
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(bx, by, br * 2.1, 0, 7);
    ctx.fill();

    for (const l of lines) {
      if (l.x1 < left || l.x0 > right) continue;
      ctx.save();
      if (l.turns) {
        ctx.translate(l.turns.x * T, -l.turns.z * T);
        ctx.rotate(l.turns.angle);
        ctx.translate(-l.turns.x * T, l.turns.z * T);
      }
      LINE_BANDS.forEach((band, i) => {
        ctx.strokeStyle = `rgba(${LINE_RGB},${band.alpha})`;
        ctx.lineWidth = band.width;
        ctx.stroke(l.bands[i]);
      });
      if (!reduced) {
        ctx.strokeStyle = `rgba(${LINE_RGB},0.55)`;
        ctx.lineWidth = 2;
        ctx.lineCap = "round";
        ctx.setLineDash([5, 62]);
        ctx.lineDashOffset = -clock * 26;
        ctx.stroke(l.whole);
        ctx.setLineDash([]);
        ctx.lineCap = "butt";
      }
      ctx.restore();
    }
  }

  function drawSolid(q) {
    const x = q.x0 * T, y = -q.z1 * T, w = (q.x1 - q.x0) * T, h = (q.z1 - q.z0) * T;
    if (q.kind === "coil") return;
    if (q.kind === "post") {
      ctx.fillStyle = INK;
      ctx.fillRect(x, y, w, h);
      return;
    }
    if (q.kind === "block") {
      ctx.fillStyle = q.used ? "#6d6c80" : INK;
      ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = q.used ? "#8a8998" : GOLD;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x + 2, y + 2, w - 4, h - 4);
      if (!q.used) {
        ctx.fillStyle = GOLD;
        ctx.font = "800 32px system-ui, -apple-system, 'Segoe UI', sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("?", x + w / 2, y + h / 2 + 2);
      }
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
    if (q.kind === "crate") return;
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

  function poleGradient(x, y0, y1, up, middle = 0.5) {
    const g = ctx.createLinearGradient(x, y0, x, y1);
    const a = up ? NORTH : SOUTH, b = up ? SOUTH : NORTH;
    g.addColorStop(0, a);
    g.addColorStop(middle - 0.08, a);
    g.addColorStop(middle, MIDDLE);
    g.addColorStop(middle + 0.08, b);
    g.addColorStop(1, b);
    return g;
  }

  /* a cuboid in oblique view: the front is the slice the hero plays in */
  function drawCuboid(m) {
    const w = m.dim[0] * T, h = m.dim[2] * T, up = m.J[2] > 0;
    const d = Math.min(1, m.dim[1] * 0.28) * T, dx = d * 0.75, dy = d * 0.6;
    const x = (m.x - m.dim[0] / 2) * T, y = -(m.z + m.dim[2] / 2) * T;
    ctx.fillStyle = up ? "#f45454" : "#2fce76";
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + w, y);
    ctx.lineTo(x + w + dx, y - dy);
    ctx.lineTo(x + dx, y - dy);
    ctx.closePath();
    ctx.fill();
    /* the side, sheared so that its bands run back from the front's along the same magnet */
    ctx.save();
    ctx.transform(1, -dy / dx, 0, 1, x + w, y);
    ctx.fillStyle = poleGradient(0, 0, h, up);
    ctx.fillRect(0, 0, dx, h);
    ctx.fillStyle = "rgba(0,0,0,0.28)";
    ctx.fillRect(0, 0, dx, h);
    ctx.restore();
    ctx.fillStyle = poleGradient(x, y, y + h, up);
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = "rgba(42,41,57,0.5)";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x, y, w, h);
  }

  function drawCylinder(m) {
    const [d, h] = m.dim, w = d * T, rx = w / 2, ry = rx * 0.26;
    const x = (m.x - d / 2) * T, y = -(m.z + h / 2) * T, H_ = h * T, up = m.J[2] > 0;
    ctx.fillStyle = poleGradient(x, y, y + H_, up);
    ctx.fillRect(x, y, w, H_);
    /* rounded sides are what make it read as a cylinder */
    const g = ctx.createLinearGradient(x, 0, x + w, 0);
    g.addColorStop(0, "rgba(0,0,0,0.42)");
    g.addColorStop(0.3, "rgba(255,255,255,0.12)");
    g.addColorStop(0.55, "rgba(255,255,255,0)");
    g.addColorStop(1, "rgba(0,0,0,0.42)");
    ctx.fillStyle = g;
    ctx.fillRect(x, y, w, H_);
    ctx.fillStyle = up ? "#f45454" : "#2fce76";
    ctx.beginPath();
    ctx.ellipse(x + rx, y, rx, ry, 0, 0, 7);
    ctx.fill();
    ctx.strokeStyle = "rgba(42,41,57,0.45)";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  function drawSphere(x, y, r, angle) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.fillStyle = poleGradient(0, -r, r, true);
    ctx.beginPath();
    ctx.arc(0, 0, r, 0, 7);
    ctx.fill();
    ctx.restore();
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 7);
    const g = ctx.createRadialGradient(x - r * 0.35, y - r * 0.4, r * 0.05, x, y, r);
    g.addColorStop(0, "rgba(255,255,255,0.55)");
    g.addColorStop(0.4, "rgba(255,255,255,0)");
    g.addColorStop(0.8, "rgba(0,0,0,0.12)");
    g.addColorStop(1, "rgba(0,0,0,0.45)");
    ctx.fillStyle = g;
    ctx.fill();
  }

  function drawMagnet(m) {
    if (m.kind === "sphere") drawSphere(m.x * T, -m.z * T, (m.dim / 2) * T, m.angle);
    else if (m.kind === "cylinder") drawCylinder(m);
    else if (m.kind === "cuboid") drawCuboid(m);
  }

  /* A coil in two halves with the hero between them, so you are seen inside it */
  const coilTop = (m) => m.z + m.h / 2;

  function drawCoilBack(m) {
    const x = m.x * T, y = -coilTop(m) * T, w = RIM * T, ry = w * 0.26;
    const north = m.I > 0;
    ctx.fillStyle = "#1c1b28";
    ctx.fillRect(x - w, y, 2 * w, (coilTop(m) - GROUND) * T);
    /* the far half of the windings, and of the mouth's ring, shows through the bore */
    ctx.strokeStyle = "#7a4a26";
    ctx.lineWidth = 6;
    for (let i = 0; i < m.n; i++) {
      const z = m.z - m.h / 2 + (m.h * (i + 0.5)) / m.n;
      if (z < GROUND) continue;
      ctx.beginPath();
      ctx.ellipse(x, -z * T, w - 3, ry, 0, Math.PI, 0);
      ctx.stroke();
    }
    ctx.strokeStyle = north ? NORTH : SOUTH;
    ctx.lineWidth = 7;
    ctx.beginPath();
    ctx.ellipse(x, y, w - 3, ry, 0, Math.PI, 0);
    ctx.stroke();
  }

  function drawCoilFront(m) {
    const x = m.x * T, y = -coilTop(m) * T, w = RIM * T, ry = w * 0.26;
    const north = m.I > 0, dir = north ? 1 : -1;
    for (let i = 0; i < m.n; i++) {
      const z = m.z - m.h / 2 + (m.h * (i + 0.5)) / m.n;
      if (z < GROUND) continue;
      ctx.strokeStyle = COPPER;
      ctx.lineWidth = 9;
      ctx.beginPath();
      ctx.ellipse(x, -z * T, w - 3, ry, 0, 0, Math.PI);
      ctx.stroke();
      ctx.strokeStyle = COPPER_LIT;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.ellipse(x, -z * T - 2, w - 3, ry, 0, 0, Math.PI);
      ctx.stroke();
      /* the current itself: at the front of a loop it runs right when B points up */
      ctx.strokeStyle = "rgba(255,255,255,0.85)";
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 15]);
      ctx.lineDashOffset = reduced ? 0 : clock * 42 * dir;
      ctx.beginPath();
      ctx.ellipse(x, -z * T, w - 3, ry, 0, 0, Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    /* the mouth is the pole you are dealing with: red pulls green feet, green pushes them */
    ctx.strokeStyle = north ? NORTH : SOUTH;
    ctx.lineWidth = 7;
    ctx.beginPath();
    ctx.ellipse(x, y, w - 3, ry, 0, 0, Math.PI);
    ctx.stroke();
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

  /* The power-up: a Dipole, which Magpylib draws as an arrow along its moment. Chunky like a
     game icon: faceted north head, south tail, white outline, and a glow that says take me. */
  function drawDipole(m) {
    const x = m.x * T, base = -m.z * T - 2 - Math.abs(Math.sin(clock * 4.5)) * 3;
    const h = 1.2 * T, tail = 0.21 * T, wide = 0.5 * T, headH = 0.62 * T;
    const neck = base - (h - headH), tip = base - h, mid = (base + tip) / 2;

    const pulse = reduced ? 0.5 : 0.5 + 0.5 * Math.sin(clock * 4);
    const halo = ctx.createRadialGradient(x, mid, 4, x, mid, T * (0.95 + 0.12 * pulse));
    halo.addColorStop(0, `rgba(245,196,84,${0.5 + 0.18 * pulse})`);
    halo.addColorStop(1, "rgba(245,196,84,0)");
    ctx.fillStyle = halo;
    ctx.beginPath();
    ctx.arc(x, mid, T * 1.1, 0, 7);
    ctx.fill();

    /* the silhouette, once in white for the outline */
    const shape = () => {
      ctx.beginPath();
      ctx.moveTo(x - tail, base);
      ctx.lineTo(x - tail, neck);
      ctx.lineTo(x - wide, neck);
      ctx.lineTo(x, tip);
      ctx.lineTo(x + wide, neck);
      ctx.lineTo(x + tail, neck);
      ctx.lineTo(x + tail, base);
      ctx.closePath();
    };
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 6;
    shape();
    ctx.stroke();
    ctx.lineJoin = "miter";

    /* the tail, round like a cylinder */
    ctx.fillStyle = SOUTH;
    ctx.fillRect(x - tail, neck, 2 * tail, base - neck);
    const round = ctx.createLinearGradient(x - tail, 0, x + tail, 0);
    round.addColorStop(0, "rgba(0,0,0,0.28)");
    round.addColorStop(0.42, "rgba(255,255,255,0.3)");
    round.addColorStop(1, "rgba(0,0,0,0.28)");
    ctx.fillStyle = round;
    ctx.fillRect(x - tail, neck, 2 * tail, base - neck);

    /* the head, two flat facets meeting along the ridge */
    const facet = (side, fill) => {
      ctx.beginPath();
      ctx.moveTo(x, tip);
      ctx.lineTo(x + side * wide, neck);
      ctx.lineTo(x, neck);
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
    };
    facet(-1, NORTH);
    facet(1, "#b80d0d");
    ctx.fillStyle = "rgba(255,255,255,0.32)";
    ctx.beginPath();
    ctx.moveTo(x, tip + 5);
    ctx.lineTo(x - wide * 0.55, neck - 4);
    ctx.lineTo(x - wide * 0.2, neck - 4);
    ctx.closePath();
    ctx.fill();

    /* Magpylib's pale band between the poles */
    ctx.fillStyle = MIDDLE;
    ctx.fillRect(x - tail, neck, 2 * tail, 4);
  }

  /* The other thing to earn: a Circle, Magpylib's current loop, a copper ring with its
     current running round. As a shot it spins as it flies. */
  function drawRing(x, y, rx, ry) {
    ctx.strokeStyle = COPPER;
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.ellipse(x, y, rx, ry, 0, 0, 7);
    ctx.stroke();
    ctx.strokeStyle = COPPER_LIT;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.ellipse(x, y - 1.5, rx, ry, 0, 0, 7);
    ctx.stroke();
    ctx.strokeStyle = "rgba(255,255,255,0.9)";
    ctx.lineWidth = 2;
    ctx.setLineDash([3, 11]);
    ctx.lineDashOffset = reduced ? 0 : clock * 40;
    ctx.beginPath();
    ctx.ellipse(x, y, rx, ry, 0, 0, 7);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function drawLoopItem(m) {
    const bob = Math.abs(Math.sin(clock * 5)) * 3;
    drawRing(m.x * T, -(m.z + 0.5) * T - bob, 0.46 * T, 0.2 * T);
  }

  function drawShot(q) {
    const turn = Math.abs(Math.cos(clock * 16));
    drawRing(q.x * T, -q.z * T, 0.26 * T, Math.max(0.06, turn) * 0.26 * T);
  }

  /* The enemy: a mushroom you could build in Magpylib, a low-poly cap like a TriangularMesh on
     a short Cylinder, north up. Hit by a fired loop it tumbles off. Stomped, its triangles
     fly apart and the stem drops. */
  const CAP_SHADES = ["rgba(0,0,0,0.3)", "rgba(0,0,0,0.16)", "rgba(0,0,0,0.04)", "rgba(255,255,255,0.1)", "rgba(255,255,255,0.2)", "rgba(0,0,0,0.12)"];

  function drawWalker(n) {
    const x = n.x * T, y = -n.z * T;
    const apart = n.dead && !n.fly;
    if (apart && n.dead < 0.02) return;
    const u = apart ? 1 - n.dead / Sim.TRAP_TIME : 0;
    const wob = n.dead ? 0 : Math.sin(clock * 9 + n.x) * 2.5;
    const r = 21, capH = 20, stemW = 25, stemH = 18, feetH = 4;
    const base = y - feetH, stemTop = base - stemH, capBase = stemTop + 3;
    ctx.save();
    if (apart) ctx.globalAlpha = Math.min(1, n.dead / 0.2);
    if (n.fly) {
      ctx.translate(x, y - 18);
      ctx.rotate(n.fly.spin);
      ctx.translate(-x, -(y - 18));
    }

    if (!apart) {
      ctx.fillStyle = INK;
      ctx.beginPath();
      ctx.ellipse(x - 8 + wob, y - 2.5, 7, 3.5, 0, 0, 7);
      ctx.ellipse(x + 8 - wob, y - 2.5, 7, 3.5, 0, 0, 7);
      ctx.fill();
    }

    /* the stem, round like a cylinder */
    const sx = x - stemW / 2, sy = stemTop + u * u * 26;
    const shade = ctx.createLinearGradient(sx, 0, sx + stemW, 0);
    shade.addColorStop(0, "rgba(0,0,0,0.34)");
    shade.addColorStop(0.42, "rgba(255,255,255,0.16)");
    shade.addColorStop(1, "rgba(0,0,0,0.34)");
    ctx.fillStyle = SOUTH;
    ctx.fillRect(sx, sy, stemW, stemH);
    ctx.fillStyle = shade;
    ctx.fillRect(sx, sy, stemW, stemH);

    /* the cap, a fan of flat triangles */
    const point = (k) => {
      const a = (Math.PI * k) / CAP_SHADES.length;
      return [x - r * Math.cos(a), capBase - capH * Math.sin(a)];
    };
    CAP_SHADES.forEach((tone, k) => {
      const [x0, y0] = point(k), [x1, y1] = point(k + 1);
      const mid = (Math.PI * (k + 0.5)) / CAP_SHADES.length;
      ctx.save();
      ctx.translate(-Math.cos(mid) * u * 50, -Math.sin(mid) * u * 42 + u * u * 34);
      if (u) {
        const cx = (x + x0 + x1) / 3, cy = (capBase + y0 + y1) / 3;
        ctx.translate(cx, cy);
        ctx.rotate((k - 2.5) * u * 1.6);
        ctx.translate(-cx, -cy);
      }
      ctx.beginPath();
      ctx.moveTo(x, capBase);
      ctx.lineTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.closePath();
      ctx.fillStyle = NORTH;
      ctx.fill();
      ctx.fillStyle = tone;
      ctx.fill();
      ctx.restore();
    });

    if (!apart) {
      /* Magpylib's pale band between the poles, under the cap */
      ctx.fillStyle = MIDDLE;
      ctx.beginPath();
      ctx.roundRect(x - r, capBase - 1, 2 * r, 4, 2);
      ctx.fill();
      /* the face: cross, and looking where it is going */
      const eyeY = stemTop + 9;
      for (const side of [-1, 1]) {
        const ex = x + side * 5.5;
        ctx.fillStyle = "#fff";
        ctx.beginPath();
        ctx.ellipse(ex, eyeY, 4, 4.6, 0, 0, 7);
        ctx.fill();
        ctx.fillStyle = INK;
        ctx.beginPath();
        ctx.arc(ex + n.dir * 1.3, eyeY + 0.6, 2.2, 0, 7);
        ctx.fill();
        ctx.strokeStyle = INK;
        ctx.lineWidth = 2.6;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(ex + side * 5, eyeY - 6.5);
        ctx.lineTo(ex - side * 3, eyeY - 3);
        ctx.stroke();
        ctx.lineCap = "butt";
      }
    }
    ctx.restore();
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

  const BLOCK = SOLIDS.find((q) => q.item === "dipole"), LOOP_BLOCK = SOLIDS.find((q) => q.item === "loop");
  const FIRE_NAME = touchUI ? "Fire" : "X";
  const FLIP_NAME = touchUI ? "Flip" : "F";
  const HINTS = [
    [16, () => {
      if (s.bird.mode === "logo") return touchUI ? "Walk or jump to get going." : "\u2190 \u2192 walk, Space jumps.";
      return "Thief! Get our letters back.";
    }],
    [34, () => {
      if (s.dipole) return "Grab the arrow: a dipole makes your magnet stronger.";
      if (!BLOCK.used && s.p.x < 23) return "Bump the ? block from below.";
      return "Land on it! Touching it any other way, it grabs you.";
    }],
    [50, () => `Green feet stick to red: north attracts south. Stuck? ${FLIP_NAME} swaps your poles.`],
    [59, () => "Green feet, jump: your red top sticks to the green underside."],
    [81, () => `Walk across, then ${FLIP_NAME} over the far ground to let go.`],
    [90, () => {
      if (s.p.fire) return `${FIRE_NAME} fires a current loop. A hit sends them flying.`;
      if (!LOOP_BLOCK.used) return "Land on it, then bump the ? block.";
      return s.loop ? "Grab the loop: a current you can fire." : "";
    }],
    [97, () => `A red mouth sucks in green feet. Inside, ${FLIP_NAME}: red on red fires you out.`],
    [134, () => `Same colours push: hover, jump off the field, and ${FLIP_NAME} for the green one.`],
    [153, () => `Green feet: this coil swallows you. ${FLIP_NAME} if you're red.`],
    [164, () => "A green mouth throws green feet back out."],
    [176, () => `It turns to follow you. Land on the red top with green feet, then ${FLIP_NAME}.`],
    [Infinity, () => (s.drop ? "It dropped the M! Grab it." : "There's the magpie.")],
  ];

  function wrapText(text, maxWidth) {
    const out = [];
    let line = "";
    for (const word of text.split(" ")) {
      const next = line ? `${line} ${word}` : word;
      if (ctx.measureText(next).width > maxWidth && line) {
        out.push(line);
        line = word;
      } else {
        line = next;
      }
    }
    out.push(line);
    return out;
  }

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
    const rows = wrapText(hintShown, VW - 56);
    const w = Math.max(...rows.map((r) => ctx.measureText(r).width)) + 36, h = 14 + rows.length * 22;
    roundRect(VW / 2 - w / 2, 54, w, h, 16);
    ctx.fillStyle = "rgba(42,41,57,0.92)";
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    rows.forEach((r, i) => ctx.fillText(r, VW / 2, 54 + 18 + i * 22));
    ctx.restore();
  }

  function slot(i) {
    return { x: 22 + i * 21, y: 27 };
  }

  function drawHUD() {
    /* on a plate, or the letters vanish whenever a dark wall scrolls behind them */
    roundRect(8, 12, 232, 30, 8);
    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.fill();
    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    ctx.font = "800 22px system-ui, -apple-system, 'Segoe UI', sans-serif";
    [..."MAGPYLIB"].forEach((ch, i) => {
      const lit = i === 0 && s.phase === "won" && winT > 0.8;
      ctx.fillStyle = lit ? RED : "rgba(42,41,57,0.28)";
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
    /* on an upright phone there is only room for the number, or not even that */
    const narrow = VW < 660, py = 12;
    if (VW > 480) {
      const w = narrow ? 118 : 180, px = VW - 82 - w;
      roundRect(px, py, w, 30, 8);
      ctx.fillStyle = "rgba(255,255,255,0.85)";
      ctx.fill();
      ctx.fillStyle = INK;
      ctx.textAlign = "left";
      ctx.fillText(`|B| ${b.toFixed(3).padStart(5)} T`, px + 8, py + 15.5);
      if (!narrow) {
        const bar = Math.max(0, Math.min(1, (Math.log10(b + 1e-9) + 2.6) / 2.4));
        ctx.fillStyle = "rgba(42,41,57,0.12)";
        ctx.fillRect(px + 124, py + 11, 46, 8);
        ctx.fillStyle = BLUE;
        ctx.fillRect(px + 124, py + 11, 46 * bar, 8);
      }
    }

    /* what your feet are right now: that decides who pulls and who pushes */
    const feetNorth = s.p.pol < 0;
    roundRect(VW - 74, py, 62, 30, 8);
    ctx.fillStyle = feetNorth ? NORTH : SOUTH;
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.textAlign = "center";
    ctx.fillText(feetNorth ? "feet N" : "feet S", VW - 43, py + 15.5);

    /* scale bar, where no thumb is */
    if (touchUI) return;
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
    ctx.fillText("Mind the magpie.", VW / 2, 130);
    if (s.phase === "title" && Math.floor(clock * 1.6) % 2 === 0) {
      ctx.font = "700 22px system-ui, -apple-system, 'Segoe UI', sans-serif";
      ctx.fillStyle = "#fff";
      ctx.fillText(touchUI ? "Tap to play" : "Press Space to start", VW / 2, VH - 42);
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
    const tz0 = titleZoom();
    view.zoom = tz0 + (1 - tz0) * e;
    const halfW = VW / (T * view.zoom) / 2;

    /* Like Mario's camera: the hero walks freely inside a window left of centre, so
       turning round moves nothing, and the view only scrolls when pushed at its edges.
       No easing and no looking ahead: either would swing the world on every turn. */
    const playHalf = VW / T / 2, windowL = -0.55 * playHalf, windowR = -0.05 * playHalf;
    if (s.phase === "intro" || title) {
      view.x = 13 + (Math.max(p.x - (windowL + windowR) / 2, 13.34) - 13) * e;
      view.z = 0.6 * (1 - e);
    } else {
      view.x = Math.max(p.x - windowR, Math.min(p.x - windowL, view.x));
      const tz = Math.max(0, p.z + p.h + 2 - VH / T);
      view.z += (tz - view.z) * (1 - Math.exp(-dt * 6));
    }
    view.x = Math.max(-6 + halfW, Math.min(LEVEL_W - halfW, view.x));
    if (title) view.x = 13;
  }

  /* ---------- one frame ---------- */

  function draw(dt) {
    const scale = canvas.width / VW;
    const b = viewBox();
    ctx.setTransform(scale * view.zoom, 0, 0, scale * view.zoom, 0, 0);
    ctx.translate(-b.left * T, b.top * T);

    drawBackdrop(b.left, b.right, b.bottom, b.top);
    for (const cp of CHECKPOINTS) drawCheck(cp);
    const seen = (m) => m.x + m.half + 1 > b.left && m.x - m.half - 1 < b.right;
    for (const m of MAGNETS) if (seen(m)) drawMagnet(m);
    /* behind the solids, so that it rises out of its block */
    if (s.dipole) drawDipole(s.dipole);
    if (s.loop) drawLoopItem(s.loop);
    for (const q of SOLIDS) {
      if (!q.magnet && q.x1 > b.left && q.x0 < b.right) drawSolid(q);
    }
    for (const m of MAGNETS) if (m.kind === "coil" && seen(m)) drawCoilBack(m);
    for (const c of s.sensors) if (!c.got && c.x > b.left - 1 && c.x < b.right + 1) drawSensor(c);
    for (const n of s.walkers) if (n.x > b.left - 2 && n.x < b.right + 2) drawWalker(n);
    drawLetters();
    drawBird();
    for (const q of s.shots) drawShot(q);
    drawHero(dt);
    if (s.p.grab) drawWalker(s.p.grab.n);
    for (const m of MAGNETS) if (m.kind === "coil" && seen(m)) drawCoilFront(m);
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
    Space: "jump", ArrowUp: "jump", KeyW: "jump", KeyF: "flip", KeyX: "fire",
  };

  function playAgain() {
    if (s.phase !== "won" || winT < WON_WAIT) return;
    s = Sim.create();
    winT = 0;
    Object.assign(view, { x: 13, z: 0.6, zoom: titleZoom() });
    syncFlip();
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
    input.left = input.right = input.jump = input.flip = input.fire = false;
  });
  canvas.addEventListener("pointerdown", () => {
    /* the game first, so that nothing about full screen can stand in its way */
    playAgain();
    wake();
    if (touchUI && !full) setFull(true);
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

  /* ---------- full screen ----------
     An iPhone has no fullscreen API for pages, so full screen here means the frame
     covering the whole window. Where the real thing exists it is asked for as well. */

  const frameEl = window.frameElement;
  const outerDoc = frameEl ? frameEl.ownerDocument : document;
  const fullBtn = document.getElementById("full");
  let full = false;

  function setFull(on) {
    full = on;
    document.body.classList.toggle("full", on);
    if (frameEl) frameEl.classList.toggle("mag404__game--full", on);
    outerDoc.documentElement.classList.toggle("mag404-playing", on);
    fullBtn.textContent = on ? "Exit" : "Full screen";
    fullBtn.blur();
    const target = frameEl || document.documentElement;
    /* the frame covering the window already did the job, the real thing is a bonus */
    try {
      if (on && target.requestFullscreen) target.requestFullscreen()?.catch(() => {});
      if (!on && outerDoc.fullscreenElement) outerDoc.exitFullscreen()?.catch(() => {});
    } catch {
      /* some browsers refuse it outright */
    }
    resize();
  }

  fullBtn.addEventListener("click", () => setFull(!full));
  /* Esc leaves real fullscreen without asking us */
  outerDoc.addEventListener("fullscreenchange", () => {
    if (!outerDoc.fullscreenElement && full && !touchUI) setFull(false);
  });

  const flipBtn = document.querySelector('[data-key="flip"]');
  const fireBtn = document.querySelector('[data-key="fire"]');
  function syncFlip() {
    flipBtn.style.background = s.p.pol > 0 ? SOUTH : NORTH;
    fireBtn.hidden = !s.p.fire;
  }

  function resize() {
    const ratio = Math.min(devicePixelRatio || 1, 2);
    const box = canvas.getBoundingClientRect();
    if (!box.width || !box.height) return;
    canvas.width = Math.round(box.width * ratio);
    canvas.height = Math.round(box.height * ratio);
    VW = (VH * box.width) / box.height;
  }
  addEventListener("resize", resize);
  new ResizeObserver(resize).observe(canvas);

  /* ---------- the loop ---------- */

  let last = 0, acc = 0;

  function consumeEvents() {
    for (const ev of s.events) {
      if (SOUND[ev]) SOUND[ev]();
      const p = s.p, cx = p.x, cz = p.z + p.h / 2;
      if (ev === "sensor") burst(cx, cz, GOLD, 7);
      else if (ev === "flip") {
        burst(cx, p.z, p.pol > 0 ? SOUTH : NORTH, 10);
        syncFlip();
      } else if (ev === "hurt") syncFlip();
      else if (ev === "stomp") burst(cx, p.z, GOLD, 12);
      else if (ev === "grab") burst(cx, p.z, NORTH, 8);
      else if (ev === "blast") burst(cx + p.face * 1.5, p.z + 0.5, NORTH, 12);
      else if (ev === "grow" || ev === "warp" || ev === "shrink") {
        burst(cx, cz, GOLD, 14);
        syncFlip();
      }
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
        fire: input.fire || tapped.fire,
      });
      tapped.jump = tapped.flip = tapped.fire = false;
      acc -= Sim.STEP;
    }
    consumeEvents();
    if (s.phase === "won") winT += dt;
    updateCamera(dt);
    draw(dt);
    requestAnimationFrame(frame);
  }

  resize();
  syncFlip();
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
