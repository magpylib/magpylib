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
  const BIG = 1.2, BIG_K = 1.5; /* the mushroom: a bigger magnet has a bigger moment */
  const G = 38, JUMP_V = 18, RUN = 9.5;
  const RUN_ACC = 70, RUN_FRIC = 60, AIR_ACC = 38;
  const K = 870; /* |m| / mass, in tile^2 / s^2 per tesla */
  const FIELD_MAX = 220; /* the force is capped so no corner can fling the hero */
  const FALL_MAX = 34, DE = 0.05, REACH = 26;
  const TILT_FOLLOW = 0.7, TILT_MAX = 1.05, TILT_FLAT = 0.3, TILT_RATE = 9, TILT_B0 = 0.015;
  const CAPSULE = 0.75; /* against a sphere the hero is rounded, so its feet meet the curve */
  const GROUND = 3, TOP = 11.5, LEVEL_W = 208;
  const POST = 190, WARP_WAIT = 0.3, RIM_SLIDE = 5;

  const JN42 = 1.3; /* tesla, an N42 grade magnet */
  /* a coil is drawn with 8 bands, each standing for 50 turns of 50 A */
  const COIL_I = 2500, COIL_D = 3.2, COIL_H = 4, COIL_N = 8, BORE = 1.3, RIM = 1.9;
  const coil = (x, I, to) => ({
    kind: "coil", x, z: GROUND - 0.05, dim: COIL_D, h: COIL_H, n: COIL_N, I, to, half: 2,
  });
  const box = (x0, x1, z0, z1, kind) => ({ x0, x1, z0, z1, kind: kind || "ground" });

  const LAUNCH = coil(92, COIL_I);
  const EXIT = coil(156, -COIL_I);
  const ENTRY = coil(141, COIL_I, EXIT);
  const COILS = [LAUNCH, ENTRY, EXIT];

  const MAGNETS = [
    { kind: "cuboid", x: 45, z: 4.5, dim: [6, 4, 3], J: [0, 0, JN42], half: 3 },
    { kind: "cuboid", x: 68, z: 11, dim: [30, 3, 2], J: [0, 0, JN42], half: 15 },
    { kind: "cylinder", x: 110, z: 2.75, dim: [6, 7.5], J: [0, 0, JN42], half: 3 },
    { kind: "cylinder", x: 119, z: 2.75, dim: [6, 7.5], J: [0, 0, JN42], half: 3 },
    { kind: "cylinder", x: 128, z: 2.75, dim: [6, 7.5], J: [0, 0, JN42], half: 3 },
    { kind: "sphere", x: 170, z: 6, dim: 6, J: [0, 0, JN42], half: 3 },
    ...COILS,
  ];

  const BLOCK = { ...box(19.5, 20.9, 6.5, 7.9, "block"), used: false };
  const SOLIDS = [
    box(-6, 33, 0, GROUND), box(36, 56, 0, GROUND), box(80, 104, 0, GROUND),
    box(132, LEVEL_W, 0, GROUND),
    box(24.5, 26.5, GROUND, GROUND + 2, "crate"),
    box(97, 104, GROUND, 11, "tower"), /* only the coil gets you up here */
    box(146, 152, GROUND, 40, "tower"), /* only the warp gets you past this */
    box(161, 164, GROUND, GROUND + 3, "crate"),
    box(176, LEVEL_W, GROUND, TOP, "tower"),
    box(LEVEL_W, LEVEL_W + 4, 0, 40, "tower"),
    box(-8, -6, 0, 40, "tower"),
    box(POST - 0.5, POST + 0.5, TOP, TOP + 2, "post"),
    BLOCK,
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

  const CHECKPOINTS = [
    { x: 38, z: GROUND }, { x: 84, z: GROUND }, { x: 100, z: 11 }, { x: 134, z: GROUND }, { x: 153, z: GROUND },
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
    ...arc(156, 10, 3, 2, 1.5), /* out of the warp */
    ...arc(170, 12.5, 3, 2.5, 0.5), /* above the sphere */
    ...arc(179, TOP + 1.2, 5, 2, 0),
  ].map((c) => ({ ...c, got: false }));

  const WALKERS = () => [
    { x: 30, z: GROUND, min: 27.8, max: 32.3, dir: -1, dead: 0 },
    { x: 87, z: GROUND, min: 85.5, max: 88.5, dir: 1, dead: 0 },
    { x: 184, z: TOP, min: 178, max: 188.5, dir: -1, dead: 0 },
  ];

  /* ---------- the field the hero feels ---------- */

  const tmp = [0, 0, 0];
  function fieldAt(x, z) {
    tmp[0] = tmp[1] = tmp[2] = 0;
    for (const m of MAGNETS) {
      if (Math.abs(x - m.x) < REACH + m.half) root.MagField.addB(m, x, 0, z, tmp);
    }
    return tmp;
  }

  function absB(x, z) {
    const b = fieldAt(x, z);
    return Math.hypot(b[0], b[1], b[2]);
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
    BLOCK.used = false;
    return {
      phase: "title", /* title, intro, play, won */
      t: 0,
      p: {
        x: START_X, z: GROUND, vx: 0, vz: 0, pol: 1, face: 1, w: W, h: H, big: false,
        grounded: false, coyote: 0, buffer: 0, cut: false, hang: false, onRail: false, rim: null,
        walk: 0, land: 0, flipT: 0, safe: 0, warp: 0, tilt: 0, pivot: 0, onSphere: false,
        a: { x: 0, z: 0 },
      },
      prev: { jump: false, flip: false },
      check: { x: START_X, z: GROUND },
      sensors: SENSORS(),
      got: 0,
      walkers: WALKERS(),
      shroom: null,
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
    const b = s.bird;
    b.mode = "fly";
    b.x = START_X - 0.65;
    b.z = GROUND + 2.3;
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
      if (p.vx > 0) p.x = q.x0 - p.w / 2;
      else if (p.vx < 0) p.x = q.x1 + p.w / 2;
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
        if (q === BLOCK && !BLOCK.used) {
          BLOCK.used = true;
          s.shroom = { x: (q.x0 + q.x1) / 2, z: q.z1 - 0.9, vx: 0, vz: 0, rise: 0.6 };
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
  }

  function respawn(s) {
    const p = s.p;
    p.x = s.check.x;
    p.z = s.check.z;
    p.vx = p.vz = 0;
    p.pol = 1;
    p.hang = false;
    p.safe = 1.5;
    p.warp = 0;
    p.tilt = 0;
    s.deaths++;
    s.events.push("hurt");
  }

  function stepShroom(s, dt) {
    const m = s.shroom, p = s.p;
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
      if (m.z < -3) s.shroom = null;
    }
    if (s.shroom && Math.abs(m.x - p.x) < p.w / 2 + 0.45 && m.z < p.z + p.h && m.z + 0.9 > p.z) {
      s.shroom = null;
      setBig(p, true);
      s.events.push("grow");
    }
  }

  function stepHero(s, input, dt) {
    const p = s.p;

    if (input.flip && !s.prev.flip) {
      p.pol = -p.pol;
      p.flipT = 0.28;
      s.events.push("flip");
    }
    if (input.jump && !s.prev.jump) p.buffer = 0.12;
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
      if (n.dead) continue;
      if (Math.abs(n.x - p.x) < p.w / 2 + 0.4 && p.z < n.z + 0.9 && p.z + p.h > n.z) {
        if (p.vz < 0 && p.z > n.z + 0.25) {
          n.dead = 0.5;
          p.vz = 13;
          s.events.push("stomp");
        } else if (!p.safe) {
          if (p.big) {
            setBig(p, false);
            p.safe = 1.5;
            s.events.push("shrink");
          } else {
            respawn(s);
            return;
          }
        }
      }
    }
  }

  /* ---------- the magpie and its bag of stolen letters ---------- */

  /* The loot bag: hanging from the claws in flight, set down beside the magpie on a perch. */
  function bagAt(b) {
    return b.mode === "perched" ? [b.x - 0.75, b.z + 0.34] : [b.x + Math.sin(b.swing) * 0.62, b.z - 0.5 - Math.cos(b.swing) * 0.62];
  }

  function stepBird(s, dt) {
    const b = s.bird, p = s.p;
    b.flap += dt;
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
    if (s.phase === "intro" && s.t > 1.7) s.phase = "play";
    if (s.phase === "title" || s.phase === "won") {
      s.bird.flap += dt;
      return;
    }
    stepBird(s, dt);
    if (s.phase === "play") stepHero(s, input, dt);
    stepShroom(s, dt);
    for (const n of s.walkers) {
      if (n.dead) {
        n.dead = Math.max(1e-9, n.dead - dt);
        continue;
      }
      n.x += n.dir * 2.2 * dt;
      if (n.x < n.min || n.x > n.max) n.dir = -n.dir;
    }
    s.prev.jump = input.jump;
    s.prev.flip = input.flip;

    const m = s.drop, p = s.p;
    if (m && Math.abs(m.x - p.x) < p.w / 2 + 0.5 && m.z < p.z + p.h + 0.5 && m.z + 1 > p.z) {
      s.phase = "won";
      s.events.push("win");
    }
  }

  root.MagrioSim = {
    create, start, step, absB, bagAt, STEP, W, H, BIG, GROUND, TOP, LEVEL_W, K, BORE, RIM,
    MAGNETS, SOLIDS, CHECKPOINTS, PERCHES, START_X,
  };
})(globalThis);
