/*
 * Super Magrio: the level and the game rules, with no drawing and no browser, so
 * a bot can play it from node.
 *
 * The world is the y = 0 slice through a real magnet scene: x runs right, z runs
 * up, and one tile is one centimetre. Magnets and coils are centred on that plane
 * and point along z, so B lies in the plane too and the field lines drawn on screen
 * are the true ones.
 *
 * The hero is a magnetic moment m at its own centre, along its body: pol = +1 has a
 * north (red) top and south (green) feet, and flipping swaps them. Its force is the
 * textbook F = grad(m . B), so north attracts south and repels north without any rule
 * saying so. The torque m x B tilts the body towards the local field line, like a
 * compass needle. What is tuned for play is |m|, standing in for mass (K below), and
 * one liberty is taken: the tilt is partial and it never turns a magnet right over,
 * where a free one that is being repelled would spin round and attract instead.
 */
(function (root) {
  "use strict";

  const STEP = 1 / 120;
  const W = 1.9, H = 2.3; /* tiles, the logo magnet is 2 wide */
  const BIG = 1.2, BIG_K = 1.5; /* the Dipole power-up: more moment, and a bigger magnet to carry it */
  const G = 38, JUMP_V = 18, RUN = 9.5;
  const RUN_ACC = 70, RUN_FRIC = 60, AIR_ACC = 38;
  const K = 870; /* |m| / mass, in tile^2 / s^2 per tesla */
  const FIELD_MAX = 220; /* the force is capped so no corner can fling the hero */
  const FALL_MAX = 34, DE = 0.05, REACH = 26;
  const TILT_FOLLOW = 0.7, TILT_MAX = 1.05, TILT_FLAT = 0.3, TILT_RATE = 9, TILT_B0 = 0.015;
  const CAPSULE = 0.75; /* against a sphere the hero is rounded, so its feet meet the curve */
  const GROUND = 3, TOP = 11.5, LEVEL_W = 208;
  const POST = 190, WARP_WAIT = 0.3, RIM_SLIDE = 5;
  const WALKER_K = 60, WALKER_PULL = 3.5, TRAP_TIME = 0.55, GRAB_TIME = 0.45, CLAMP_TIME = 0.13, CLAMP_DROP = 16;
  const SPHERE_K = 900, SPHERE_DAMP = 6, SPHERE_SPIN_MAX = 3, SPHERE_REACH = 12, SPHERE_MAX = 1.2;
  const SHOT_V = 15, SHOT_G = 34, SHOT_BOUNCE = 9, SHOT_LIFE = 1.6, MAX_SHOTS = 2;

  const JN42 = 1.3; /* tesla, an N42 grade magnet */
  /* a coil is drawn with 8 bands, each standing for 50 turns of 50 A */
  const COIL_I = 2500, COIL_D = 3.2, COIL_H = 4, COIL_N = 8, BORE = 1.3, RIM = 1.9;
  const coil = (x, I, to) => ({
    kind: "coil", x, z: GROUND - 0.05, dim: COIL_D, h: COIL_H, n: COIL_N, I, to, half: 2,
  });
  const box = (x0, x1, z0, z1, kind) => ({ x0, x1, z0, z1, kind: kind || "ground" });

  const LAUNCH = coil(92, COIL_I);
  const EXIT = coil(157, -COIL_I);
  const ENTRY = coil(141, COIL_I, EXIT);
  const COILS = [LAUNCH, ENTRY, EXIT];

  const MAGNETS = [
    { kind: "cuboid", x: 45, z: 4.5, dim: [6, 4, 3], J: [0, 0, JN42], half: 3 },
    { kind: "cuboid", x: 68, z: 11, dim: [30, 3, 2], J: [0, 0, JN42], half: 15 },
    { kind: "cylinder", x: 110, z: 2.75, dim: [6, 7.5], J: [0, 0, JN42], half: 3 },
    { kind: "cylinder", x: 119, z: 2.75, dim: [6, 7.5], J: [0, 0, -JN42], half: 3 },
    { kind: "cylinder", x: 128, z: 2.75, dim: [6, 7.5], J: [0, 0, JN42], half: 3 },
    { kind: "sphere", x: 171, z: 6, dim: 6, J: [0, 0, JN42], half: 3 },
    ...COILS,
  ];

  /* ? blocks, each holding one thing to earn */
  const BLOCK = { ...box(19.5, 20.9, 6.5, 7.9, "block"), item: "dipole", used: false };
  const LOOP_BLOCK = { ...box(85.8, 87.2, 6.5, 7.9, "block"), item: "loop", used: false };
  const SOLIDS = [
    box(-6, 33, 0, GROUND), box(36, 56, 0, GROUND), box(80, 104, 0, GROUND),
    box(132, LEVEL_W, 0, GROUND),
    box(24.5, 26.5, GROUND, GROUND + 2, "crate"),
    box(97, 104, GROUND, 11, "tower"), /* only the coil gets you up here */
    box(146, 152, GROUND, 40, "tower"), /* only the warp gets you past this */
    box(162, 165, GROUND, GROUND + 3, "crate"),
    box(176, LEVEL_W, GROUND, TOP, "tower"),
    box(LEVEL_W, LEVEL_W + 4, 0, 40, "tower"),
    box(-8, -6, 0, 40, "tower"),
    box(POST - 0.5, POST + 0.5, TOP, TOP + 2, "post"),
    BLOCK, LOOP_BLOCK,
  ];
  for (const m of MAGNETS) {
    if (m.kind === "sphere") continue;
    if (m.kind === "coil") {
      const top = m.z + m.h / 2;
      SOLIDS.push({ ...box(m.x - RIM, m.x - BORE, GROUND, top, "coil"), mouth: m.x });
      SOLIDS.push({ ...box(m.x + BORE, m.x + RIM, GROUND, top, "coil"), mouth: m.x });
      continue;
    }
    const [w, h] = m.kind === "cuboid" ? [m.dim[0], m.dim[2]] : m.dim;
    SOLIDS.push({ ...box(m.x - w / 2, m.x + w / 2, m.z - h / 2, m.z + h / 2, "magnet"), magnet: m });
  }
  const SPHERES = MAGNETS.filter((m) => m.kind === "sphere");
  const SPHERE = SPHERES[0]; /* fixed in place but free to turn, see stepSphere */

  const CHECKPOINTS = [
    { x: 22.5, z: GROUND }, { x: 38, z: GROUND }, { x: 84, z: GROUND }, { x: 100, z: 11 }, { x: 134, z: GROUND }, { x: 153, z: GROUND },
  ];
  const WORD = "Magpylib";
  const START_X = 9;
  const LETTER_Z = GROUND + 0.75, LETTER_HOP = 0.42;
  const LETTER_X = [11.1, 11.93, 12.61, 13.31, 13.97, 14.44, 14.8, 15.35]; /* set like the wordmark */
  /* where the magpie waits, always the right edge of something so its tail hangs free */
  const PERCHES = [
    { x: 82.7, z: 12, near: 12 },
    { x: 130.7, z: 6.5, near: 11 },
    { x: POST + 0.2, z: TOP + 2, near: 14, last: true },
  ];

  function arc(x0, z0, n, dx, h) {
    const out = [];
    for (let i = 0; i < n; i++) {
      const t = n > 1 ? i / (n - 1) : 0.5;
      out.push({ x: x0 + i * dx, z: z0 + 4 * h * t * (1 - t) });
    }
    return out;
  }

  const SENSORS = () => [
    ...arc(16, GROUND + 1, 3, 1.2, 0),
    ...arc(27, GROUND + 3, 4, 1.8, 2),
    ...arc(43, GROUND + 7.5, 3, 2, 0), /* over the block, only a flip gets there */
    ...arc(60, 8, 5, 4, 0), /* under the rail */
    ...arc(92, 9, 2, 0, 0), /* in the coil's line of fire */
    ...arc(114.5, 12, 2, 9, 0), /* between the cylinders */
    ...arc(157, 10, 3, 2, 1.5), /* out of the warp */
    ...arc(171, 12.5, 2, 2.5, 0.5), /* above the sphere */
    ...arc(179, TOP + 1.2, 5, 2, 0),
  ].map((c) => ({ ...c, got: false }));

  /* min and max are its beat, lo and hi the ground it can be nudged along */
  const walker = (x, z, min, max, lo, hi, dir) => ({ x, z, min, max, lo, hi, dir, dead: 0, pull: 0, fly: null });
  const WALKERS = () => [
    walker(31, GROUND, 29.5, 32.3, 27.2, 32.3, -1),
    walker(86, GROUND, 84.5, 87, 81, 87.6, 1),
    walker(136, GROUND, 135, 137, 133, 137.6, 1),
    walker(184, TOP, 178, 188.5, 177, 188.9, -1),
  ];

  /* ---------- the field the hero feels ---------- */

  const tmp = [0, 0, 0];
  function fieldAt(x, z, skip) {
    tmp[0] = tmp[1] = tmp[2] = 0;
    for (const m of MAGNETS) {
      if (m !== skip && Math.abs(x - m.x) < REACH + m.half) root.MagField.addB(m, x, 0, z, tmp);
    }
    return tmp;
  }

  function absB(x, z, skip) {
    const b = fieldAt(x, z, skip);
    return Math.hypot(b[0], b[1], b[2]);
  }

  /* The sphere swivels on a mount at its centre, like a compass in a gimbal: the hero's field,
     a dipole's at this range, turns it towards the field line through it. Free to turn right
     over it would attract whatever the hero's feet, and fling the hero about, so the mount
     keeps its north pole up. */
  function setSphereAngle(a) {
    SPHERE.angle = a;
    SPHERE.J = [JN42 * Math.sin(a), 0, JN42 * Math.cos(a)];
  }

  function stepSphere(s, dt) {
    const p = s.p, m = SPHERE;
    const dx = m.x - p.x, dz = m.z - (p.z + p.h / 2), r = Math.hypot(dx, dz);
    let torque = 0;
    if (r < SPHERE_REACH) {
      const k = p.pol * (p.big ? BIG_K : 1), mx = k * Math.sin(p.tilt), mz = k * Math.cos(p.tilt);
      const ux = dx / r, uz = dz / r, dot = 3 * (mx * ux + mz * uz), r3 = r * r * r;
      const bx = (dot * ux - mx) / r3, bz = (dot * uz - mz) / r3;
      /* the line, not the direction: twice the angle, so that it settles either way along it */
      const c = Math.cos(m.angle), n = Math.sin(m.angle);
      torque = SPHERE_K * (c * bx - n * bz) * Math.sign(c * bz + n * bx || 1);
    }
    m.spin += (torque - SPHERE_DAMP * m.spin) * dt;
    m.spin = Math.max(-SPHERE_SPIN_MAX, Math.min(SPHERE_SPIN_MAX, m.spin));
    const next = m.angle + m.spin * dt;
    if (Math.abs(next) > SPHERE_MAX) m.spin = 0;
    setSphereAngle(Math.max(-SPHERE_MAX, Math.min(SPHERE_MAX, next)));
  }

  /* B along the hero's body, m . B per unit of m */
  function alongBody(tilt, x, z) {
    const b = fieldAt(x, z);
    return Math.sin(tilt) * b[0] + Math.cos(tilt) * b[2];
  }

  /* The field line through the hero, as the tilt nearest to upright. Weak fields hardly
     turn it, and flat on the ground it can only lean, or its feet would dig in. */
  function tiltTarget(p) {
    const b = fieldAt(p.x, p.z + p.h / 2);
    const mag = Math.hypot(b[0], b[2]);
    if (mag === 0) return 0;
    let line = Math.atan2(b[0], b[2]);
    if (line > Math.PI / 2) line -= Math.PI;
    else if (line < -Math.PI / 2) line += Math.PI;
    const max = (p.grounded || p.hang) && !p.onSphere ? TILT_FLAT : TILT_MAX;
    const want = TILT_FOLLOW * line * (mag / (mag + TILT_B0));
    return Math.max(-max, Math.min(max, want));
  }

  function fieldAccel(p, out) {
    const cx = p.x, cz = p.z + p.h / 2, k = p.pol * K * (p.big ? BIG_K : 1), t = p.tilt;
    let ax = (k * (alongBody(t, cx + DE, cz) - alongBody(t, cx - DE, cz))) / (2 * DE);
    let az = (k * (alongBody(t, cx, cz + DE) - alongBody(t, cx, cz - DE))) / (2 * DE);
    const a = Math.hypot(ax, az);
    if (a > FIELD_MAX) {
      ax *= FIELD_MAX / a;
      az *= FIELD_MAX / a;
    }
    out.x = ax;
    out.z = az;
    p.attract = p.pol * alongBody(t, cx, cz) > 0; /* m along B: being drawn in, else pushed out */
  }

  /* ---------- state ---------- */

  function create() {
    BLOCK.used = LOOP_BLOCK.used = false;
    SPHERE.spin = 0;
    setSphereAngle(0);
    return {
      phase: "title", /* title, intro, play, won */
      t: 0,
      p: {
        x: START_X, z: GROUND, vx: 0, vz: 0, pol: 1, face: 1, w: W, h: H, big: false, fire: false, grab: null, clamp: null, pinch: 0,
        grounded: false, coyote: 0, buffer: 0, cut: false, hang: false, onRail: false, rim: null,
        walk: 0, land: 0, flipT: 0, safe: 0, warp: 0, tilt: 0, pivot: 0, onSphere: false,
        a: { x: 0, z: 0 },
      },
      prev: { jump: false, flip: false, fire: false },
      check: { x: START_X, z: GROUND },
      sensors: SENSORS(),
      got: 0,
      walkers: WALKERS(),
      dipole: null,
      loop: null,
      shots: [],
      deaths: 0,
      bird: { x: 0, z: 0, mode: "logo", route: [], perch: -1, face: 1, flap: 0, swing: 0, bump: 0, loot: 0 },
      letters: [...WORD].map((ch, i) => ({
        ch, i, x: LETTER_X[i], z: LETTER_Z, x0: LETTER_X[i], z0: LETTER_Z, size: 1, t: -1,
      })),
      drop: null,
      events: [],
    };
  }

  function start(s) {
    if (s.phase !== "title") return;
    s.phase = "intro";
    s.t = 0;
  }

  /* the magpie sits on the magnet's shoulder, as in the logo, until the hero first moves */
  function takeOff(s) {
    const b = s.bird, p = s.p;
    b.mode = "fly";
    b.flap = 0;
    b.x = p.x - 0.65;
    b.z = p.z + 2.3;
    b.route = [
      { x: LETTER_X[0] - 0.6, z: GROUND + 3.2, v: 9 },
      { x: LETTER_X[7] + 1.4, z: GROUND + 3.2, v: 7 },
      { x: 30, z: 14, v: 20 },
      { x: PERCHES[0].x - 3, z: 15, v: 26 },
      { ...PERCHES[0], v: 18, perch: 0 },
    ];
    s.events.push("squawk");
  }

  /* ---------- the hero ---------- */

  function hit(p, q) {
    return p.x + p.w / 2 > q.x0 && p.x - p.w / 2 < q.x1 && p.z + p.h > q.z0 && p.z < q.z1;
  }

  function move(s, p, dt) {
    p.onRail = false;
    p.onSphere = false;
    p.rim = null;
    const wasGrounded = p.grounded;
    p.grounded = false;

    p.x += p.vx * dt;
    for (const q of SOLIDS) {
      if (!hit(p, q)) continue;
      /* out by the nearer side, whichever way the hero was going */
      const toLeft = p.x + p.w / 2 - q.x0, toRight = q.x1 - (p.x - p.w / 2);
      p.x += toLeft < toRight ? -toLeft : toRight;
      p.vx = 0;
    }

    p.z += p.vz * dt;
    for (const q of SOLIDS) {
      if (!hit(p, q)) continue;
      if (p.vz <= 0) {
        p.z = q.z1;
        p.grounded = true;
        if (q.mouth !== undefined) p.rim = q.mouth;
      } else {
        p.z = q.z0 - p.h;
        if (q.magnet) p.onRail = true;
        if (q.item && !q.used) {
          q.used = true;
          s[q.item] = { x: (q.x0 + q.x1) / 2, z: q.z1 - 0.9, vx: 0, vz: 0, rise: 0.6 };
          s.events.push("bump");
        }
      }
      p.vz = 0;
    }

    for (const m of SPHERES) {
      const r = m.dim / 2 + CAPSULE;
      const qz = Math.max(p.z + CAPSULE, Math.min(m.z, p.z + p.h - CAPSULE));
      const dx = p.x - m.x, dz = qz - m.z, d = Math.hypot(dx, dz);
      if (d >= r) continue;
      const nx = d > 0 ? dx / d : 0, nz = d > 0 ? dz / d : 1;
      p.x += nx * (r - d);
      p.z += nz * (r - d);
      p.onSphere = true;
      const vn = p.vx * nx + p.vz * nz;
      if (vn < 0) {
        p.vx -= vn * nx;
        p.vz -= vn * nz;
      }
      if (nz > 0.5) p.grounded = true;
    }

    if (!p.grounded && wasGrounded && p.vz <= 0) p.coyote = 0.1;
  }

  function setBig(p, big) {
    p.big = big;
    p.w = big ? W * BIG : W;
    p.h = big ? H * BIG : H;
    /* grown into something, step out of it the shortest way */
    for (const q of SOLIDS) {
      if (!hit(p, q)) continue;
      const left = p.x + p.w / 2 - q.x0, right = q.x1 - (p.x - p.w / 2);
      const down = p.z + p.h - q.z0, up = q.z1 - p.z;
      const least = Math.min(left, right, down, up);
      if (least === left) p.x -= left;
      else if (least === right) p.x += right;
      else if (least === down) p.z -= down;
      else p.z += up;
    }
  }

  function respawn(s) {
    const p = s.p;
    p.x = s.check.x;
    p.z = s.check.z;
    p.vx = p.vz = 0;
    p.pol = 1;
    p.hang = false;
    p.grab = null;
    p.clamp = null;
    p.pinch = 0;
    p.safe = 1.5;
    p.warp = 0;
    p.tilt = 0;
    s.deaths++;
    s.events.push("hurt");
  }

  function stepItem(s, key, dt) {
    const m = s[key], p = s.p;
    if (!m) return;
    if (m.rise > 0) {
      m.rise -= dt;
      m.z += 1.5 * dt;
      if (m.rise <= 0) m.vx = 3;
    } else {
      m.vz = Math.max(-20, m.vz - G * dt);
      m.x += m.vx * dt;
      for (const q of SOLIDS) {
        if (m.x + 0.45 > q.x0 && m.x - 0.45 < q.x1 && m.z + 0.9 > q.z0 && m.z < q.z1) {
          m.vx = -m.vx;
          m.x += m.vx * dt;
        }
      }
      m.z += m.vz * dt;
      for (const q of SOLIDS) {
        if (m.x + 0.45 > q.x0 && m.x - 0.45 < q.x1 && m.z + 0.9 > q.z0 && m.z < q.z1 && m.vz <= 0) {
          m.z = q.z1;
          m.vz = 0;
        }
      }
      if (m.z < -3) s[key] = null;
    }
    if (s[key] && Math.abs(m.x - p.x) < p.w / 2 + 0.45 && m.z < p.z + p.h && m.z + 0.9 > p.z) {
      s[key] = null;
      if (key === "dipole") setBig(p, true);
      else p.fire = true;
      s.events.push("grow");
    }
  }

  function stepHero(s, input, dt) {
    const p = s.p;
    if (p.grab) {
      stepGrab(s, dt);
      return;
    }
    if (p.clamp) {
      stepClamp(s, dt);
      return;
    }
    p.pinch = Math.max(0, p.pinch - dt * 6);

    if (input.flip && !s.prev.flip) {
      p.pol = -p.pol;
      p.flipT = 0.28;
      s.events.push("flip");
    }
    if (input.jump && !s.prev.jump) p.buffer = 0.12;
    if (input.fire && !s.prev.fire && p.fire && s.shots.length < MAX_SHOTS) {
      s.shots.push({ x: p.x + p.face * p.w * 0.5, z: p.z + p.h * 0.5, vx: p.face * SHOT_V, vz: 2, life: SHOT_LIFE });
      s.events.push("shoot");
    }
    const dir = (input.right ? 1 : 0) - (input.left ? 1 : 0);
    if (dir) p.face = dir;

    fieldAccel(p, p.a);

    /* held up against a magnet from below, the hero walks along it */
    p.hang = p.onRail && p.a.z > 0.3 * G + G;

    const on = p.grounded || p.hang;
    if (dir && (Math.abs(p.vx) < RUN || dir * p.vx < 0)) {
      p.vx += dir * (on ? RUN_ACC : AIR_ACC) * dt;
    } else if (!dir && on) {
      p.vx -= Math.sign(p.vx) * Math.min(Math.abs(p.vx), RUN_FRIC * dt);
    }
    p.vx += p.a.x * dt;
    p.vz += (p.a.z - G) * dt;

    /* a rim is a funnel: stand on one without walking away and you slide in */
    if (p.rim !== null && dir * (p.x - p.rim) <= 0) p.x += Math.sign(p.rim - p.x) * RIM_SLIDE * dt;

    p.coyote = Math.max(0, p.coyote - dt);
    p.buffer = Math.max(0, p.buffer - dt);
    /* pushed from below, the field under the hero is as good as a floor to jump from */
    const cushion = p.a.z > 0.6 * G && p.vz < 4 && !p.hang;
    if (p.buffer > 0 && (p.grounded || p.coyote > 0 || cushion)) {
      p.vz = JUMP_V;
      p.buffer = p.coyote = 0;
      p.grounded = false;
      p.cut = true;
      s.events.push("jump");
    }
    /* a short tap is a short hop, unless the field is the one doing the lifting */
    if (p.cut && !input.jump && p.vz > 6 && Math.abs(p.a.z) < G / 2) {
      p.vz *= 0.55;
      p.cut = false;
    }
    if (p.vz <= 0) p.cut = false;

    p.vz = Math.max(-FALL_MAX, p.vz);
    const was = p.vz;
    move(s, p, dt);
    if (p.grounded && was < -14) p.land = 0.16;

    p.tilt += (tiltTarget(p) - p.tilt) * (1 - Math.exp(-dt * TILT_RATE));
    /* turns about whatever it is standing on or hanging from, in the air about its middle */
    const pivotTo = p.grounded ? -1 : p.hang ? 1 : 0;
    p.pivot += (pivotTo - p.pivot) * (1 - Math.exp(-dt * 14));

    p.walk += Math.abs(p.vx) * dt * 2.6;
    p.land = Math.max(0, p.land - dt);
    p.flipT = Math.max(0, p.flipT - dt);
    p.safe = Math.max(0, p.safe - dt);

    /* sucked to the bottom of a coil with a partner: out through the partner */
    const inCoil = COILS.find((c) => c.to && Math.abs(p.x - c.x) < BORE && p.z < c.z + c.h / 2);
    if (inCoil && p.grounded && p.a.z < 0) {
      p.warp += dt;
      if (p.warp > WARP_WAIT) {
        p.x = inCoil.to.x;
        p.z = GROUND;
        p.vx = p.vz = 0;
        p.warp = 0;
        s.events.push("warp");
      }
    } else {
      p.warp = 0;
    }

    if (p.z < -4) {
      respawn(s);
      return;
    }

    for (const cp of CHECKPOINTS) {
      if (p.x > cp.x && s.check.x < cp.x) {
        s.check = { x: cp.x, z: cp.z };
        s.events.push("check");
      }
    }

    for (const sensor of s.sensors) {
      if (!sensor.got && Math.abs(sensor.x - p.x) < p.w / 2 + 0.3
          && sensor.z > p.z - 0.3 && sensor.z < p.z + p.h + 0.3) {
        sensor.got = true;
        s.got++;
        s.events.push("sensor");
      }
    }

    for (const n of s.walkers) {
      if (n.dead || p.safe) continue;
      if (Math.abs(n.x - p.x) < p.w / 2 + 0.4 && p.z < n.z + 0.9 && p.z + p.h > n.z) {
        /* landing on it clamps it, any other touch and it grabs on */
        if (p.vz < 0 && p.z > n.z + 0.25) {
          p.clamp = { n, t: CLAMP_TIME };
          p.vx = p.vz = 0;
          s.events.push("clamp");
          return;
        } else {
          p.grab = { n, t: GRAB_TIME, side: Math.sign(n.x - p.x) || p.face };
          p.vx = p.vz = 0;
          s.events.push("grab");
          return;
        }
      }
    }
  }

  /* A stomp is a clamp, the hole in a horseshoe could crush nothing: the hero drops down around
     the enemy, its legs snap shut on it, it bursts, and the hero hops off with its legs open again. */
  function stepClamp(s, dt) {
    const p = s.p, c = p.clamp, n = c.n;
    c.t -= dt;
    n.x += (p.x - n.x) * Math.min(1, dt * 40);
    p.z = Math.max(n.z, p.z - CLAMP_DROP * dt);
    p.pinch = 1 - Math.max(0, c.t) / CLAMP_TIME;
    if (c.t > 0) return;
    p.clamp = null;
    n.dead = TRAP_TIME;
    p.vz = 11;
    p.grounded = false;
    s.events.push("stomp");
  }

  /* held by an enemy: a moment to see it, then it costs what was earned, one thing at a time */
  function stepGrab(s, dt) {
    const p = s.p, g = p.grab, n = g.n;
    if (n.dead) {
      p.grab = null; /* a loop already in the air knocked it off */
      p.safe = 0.6;
      return;
    }
    n.x = p.x + g.side * p.w * 0.34;
    n.z = p.z;
    g.t -= dt;
    if (g.t > 0) return;
    p.grab = null;
    if (p.fire || p.big) {
      if (p.fire) p.fire = false;
      else setBig(p, false);
      p.safe = 1.5;
      p.vz = 7;
      n.x = Math.max(n.lo, Math.min(n.hi, n.x));
      s.events.push("shrink");
    } else {
      n.x = Math.max(n.lo, Math.min(n.hi, n.x));
      respawn(s);
    }
  }

  /* ---------- the enemies: small magnets, north up, nudged by the hero's field ---------- */

  /* The force on an enemy from the hero, pole by pole: like poles push, unlike pull. Each
     magnet is its two poles, which is the same physics as its field at this range. */
  function forceOnWalker(p, n, out) {
    const m = p.pol * (p.big ? BIG_K : 1), reach = p.h / 2 - 0.3;
    const lean = Math.sin(p.tilt) * reach, up = Math.cos(p.tilt) * reach, mid = p.z + p.h / 2;
    const hero = [[p.x + lean, mid + up, m], [p.x - lean, mid - up, -m]];
    const walker = [[n.x, n.z + 0.75, 1], [n.x, n.z + 0.1, -1]];
    let fx = 0, fz = 0;
    for (const [ax, az, qa] of hero) {
      for (const [bx, bz, qb] of walker) {
        const dx = bx - ax, dz = bz - az, r = Math.max(0.5, Math.hypot(dx, dz));
        const k = (qa * qb) / (r * r * r);
        fx += k * dx;
        fz += k * dz;
      }
    }
    out.x = WALKER_K * fx;
    out.z = WALKER_K * fz;
  }

  const walkerForce = { x: 0, z: 0 };

  function stepWalker(s, n, dt) {
    if (n.fly) {
      n.fly.vz -= G * dt;
      n.x += n.fly.vx * dt;
      n.z += n.fly.vz * dt;
      n.fly.spin += n.fly.vx * dt * 1.4;
      if (n.z < -6) {
        n.fly = null;
        n.dead = 1e-9;
      }
      return;
    }
    if (n.dead) {
      n.dead = Math.max(1e-9, n.dead - dt);
      return;
    }
    const p = s.p;
    if ((p.grab && p.grab.n === n) || (p.clamp && p.clamp.n === n)) return;
    forceOnWalker(p, n, walkerForce);
    /* a nudge at its own level only: from above it would slide out from under a landing */
    const level = p.z < n.z + 1.2;
    const pull = level ? Math.max(-WALKER_PULL, Math.min(WALKER_PULL, walkerForce.x)) : 0;
    n.pull += (pull - n.pull) * (1 - Math.exp(-dt * 8));
    n.x += (n.dir * 2.2 + n.pull) * dt;
    if (n.x < n.min) n.dir = 1;
    else if (n.x > n.max) n.dir = -1;
    n.x = Math.max(n.lo, Math.min(n.hi, n.x));
  }

  /* A fired loop bounces along the ground. It sends an enemy flying, a wall ends it. */
  function stepShots(s, dt) {
    s.shots = s.shots.filter((q) => {
      q.life -= dt;
      if (q.life <= 0 || q.z < -3) return false;
      q.vz -= SHOT_G * dt;
      q.x += q.vx * dt;
      const inside = (w) => q.x + 0.25 > w.x0 && q.x - 0.25 < w.x1 && q.z + 0.25 > w.z0 && q.z - 0.25 < w.z1;
      if (SOLIDS.some(inside)) return false;
      q.z += q.vz * dt;
      for (const w of SOLIDS) {
        if (inside(w)) {
          if (q.vz > 0) return false;
          q.z = w.z1 + 0.25;
          q.vz = SHOT_BOUNCE;
        }
      }
      for (const n of s.walkers) {
        if (!n.dead && Math.abs(n.x - q.x) < 0.8 && q.z > n.z - 0.2 && q.z < n.z + 1.2) {
          n.fly = { vx: Math.sign(q.vx) * 11, vz: 12, spin: 0 };
          n.dead = 1;
          s.events.push("blast");
          return false;
        }
      }
      return true;
    });
  }

  /* ---------- the magpie and its bag of stolen letters ---------- */

  /* The loot bag: hanging from the claws in flight, set down beside the magpie on a perch. */
  function bagAt(b) {
    return b.mode === "perched" ? [b.x - 0.75, b.z + 0.34] : [b.x + Math.sin(b.swing) * 0.62, b.z - 0.5 - Math.cos(b.swing) * 0.62];
  }

  function stepBird(s, dt) {
    const b = s.bird, p = s.p;
    b.flap += dt;
    if (b.mode === "logo") {
      if (s.phase === "play" && (Math.abs(p.vx) > 0.5 || Math.abs(p.vz) > 1)) takeOff(s);
      return;
    }
    if (b.mode === "fly") {
      const r = b.route[0];
      if (!r) {
        b.mode = "gone";
        return;
      }
      const dx = r.x - b.x, dz = r.z - b.z, d = Math.hypot(dx, dz), v = r.v * dt;
      if (Math.abs(dx) > 0.01) b.face = Math.sign(dx);
      if (d <= v) {
        b.x = r.x;
        b.z = r.z;
        b.route.shift();
        if (r.perch !== undefined) {
          b.mode = "perched";
          b.perch = r.perch;
        }
      } else {
        b.x += (dx / d) * v;
        b.z += (dz / d) * v;
      }
    } else if (b.mode === "perched") {
      const here = PERCHES[b.perch];
      if (Math.abs(p.x - b.x) < here.near && s.phase === "play") {
        s.events.push("squawk");
        b.mode = "fly";
        if (here.last) {
          /* the bag gives up the first letter it took */
          const [bx, bz] = bagAt(b);
          s.drop = { x: Math.min(bx, POST - 1.2), z: bz, vx: -2.5, vz: 9, ch: WORD[0], rest: false };
          b.loot--;
          b.bump = 0.2;
          b.route = [{ x: LEVEL_W + 6, z: 22, v: 26 }];
        } else {
          b.route = [
            { x: b.x + 4, z: b.z + 3, v: 22 },
            { x: PERCHES[b.perch + 1].x - 3, z: 16, v: 27 },
            { ...PERCHES[b.perch + 1], v: 27, perch: b.perch + 1 },
          ];
        }
      }
    }

    /* the bag swings behind the way it is carried */
    const vx = b.mode === "fly" && b.route[0] ? Math.sign(b.route[0].x - b.x) * b.route[0].v : 0;
    b.swing += (-vx * 0.022 - b.swing) * (1 - Math.exp(-dt * 6));
    b.bump = Math.max(0, b.bump - dt);

    /* as the magpie passes, each letter hops up and drops into the bag */
    const [bagX, bagZ] = bagAt(b);
    for (const l of s.letters) {
      if (l.t < 0 && b.mode === "fly" && b.x > l.x0 - 0.5 && s.phase !== "title") {
        l.t = 0;
        s.events.push("pluck");
      }
      if (l.t >= 0 && l.t < 1) {
        l.t = Math.min(1, l.t + dt / LETTER_HOP);
        const e = l.t * l.t * (3 - 2 * l.t);
        l.x = l.x0 + (bagX - l.x0) * e;
        l.z = l.z0 + (bagZ - l.z0) * e + Math.sin(Math.PI * l.t) * 0.9;
        l.size = 1 - 0.75 * e;
        if (l.t === 1) {
          b.loot++;
          b.bump = 0.14;
        }
      }
    }

    if (s.drop && !s.drop.rest) {
      const m = s.drop;
      m.vz -= 26 * dt;
      m.x += m.vx * dt;
      m.z += m.vz * dt;
      if (m.z < TOP + 0.05 && m.x < LEVEL_W && m.x > 176) {
        m.z = TOP + 0.05;
        m.vx *= 0.5;
        if (Math.abs(m.vz) < 3) {
          m.vz = 0;
          m.rest = true;
        } else {
          m.vz *= -0.45;
          s.events.push("clink");
        }
      }
    }
  }

  function step(s, input) {
    const dt = STEP;
    s.t += dt;
    if (s.phase === "intro" && s.t > 0.4) s.phase = "play";
    if (s.phase === "title" || s.phase === "won") {
      s.bird.flap += dt;
      return;
    }
    stepBird(s, dt);
    if (s.phase === "play") {
      stepHero(s, input, dt);
      stepSphere(s, dt);
    }
    stepItem(s, "dipole", dt);
    stepItem(s, "loop", dt);
    stepShots(s, dt);
    for (const n of s.walkers) stepWalker(s, n, dt);
    s.prev.jump = input.jump;
    s.prev.flip = input.flip;
    s.prev.fire = !!input.fire;

    const m = s.drop, p = s.p;
    if (m && Math.abs(m.x - p.x) < p.w / 2 + 0.5 && m.z < p.z + p.h + 0.5 && m.z + 1 > p.z) {
      s.phase = "won";
      s.events.push("win");
    }
  }

  root.MagrioSim = {
    create, start, step, absB, bagAt, SPHERE, forceOnWalker, TRAP_TIME, STEP, W, H, BIG, GROUND, TOP, LEVEL_W, K, BORE, RIM,
    MAGNETS, SOLIDS, CHECKPOINTS, PERCHES, START_X,
  };
})(globalThis);
