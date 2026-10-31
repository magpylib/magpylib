/*
 * Super Magrio: the level and the game rules, with no drawing and no browser, so
 * a bot can play it from node.
 *
 * The world is the y = 0 slice through a real magnet scene: x runs right, z runs
 * up, and one tile is one centimetre. The magnets are centred on that plane and
 * magnetized in it, so B lies in the plane too and the field lines drawn on
 * screen are the true ones.
 *
 * The hero is a magnetic moment m at its own centre that stays parallel (pull) or
 * antiparallel (push) to the local field. Its force is then the textbook
 * F = grad(m . B) = +-|m| grad|B|. What is tuned for play is |m|, standing in for
 * mass (K below), and one liberty is taken: a real magnet pushed like that would
 * spin round and pull instead. The hero does not.
 */
(function (root) {
  "use strict";

  const STEP = 1 / 120;
  const W = 1.9, H = 2.3; /* tiles, the logo magnet is 2 wide */
  const G = 38, JUMP_V = 18, RUN = 9.5;
  const RUN_ACC = 70, RUN_FRIC = 60, AIR_ACC = 38, STUCK_ACC = 90, STUCK_FRIC = 140;
  const K = 870; /* |m| / mass, in tile^2 / s^2 per tesla */
  const FIELD_MAX = 220; /* the force is capped so no corner can fling the hero */
  const FALL_MAX = 34, DE = 0.05, REACH = 26;
  const GROUND = 3, TOP = 11.5, LEVEL_W = 176;

  const JN42 = 1.3; /* tesla, an N42 grade magnet */
  const box = (x0, x1, z0, z1, kind) => ({ x0, x1, z0, z1, kind: kind || "ground" });

  const MAGNETS = [
    { kind: "cuboid", name: "Cuboid", x: 45, z: 4.5, dim: [6, 4, 3], J: [0, 0, JN42] },
    { kind: "cuboid", name: "Cuboid", x: 68, z: 11, dim: [30, 3, 2], J: [0, 0, -JN42] },
    { kind: "cylinder", name: "Cylinder", x: 94, z: 1.75, dim: [4, 5.5], J: [0, 0, JN42] },
    { kind: "cylinder", name: "Cylinder", x: 102, z: 2.75, dim: [4, 7.5], J: [0, 0, JN42] },
    { kind: "cylinder", name: "Cylinder", x: 110, z: 3.75, dim: [4, 9.5], J: [0, 0, JN42] },
    { kind: "sphere", name: "Sphere", x: 130, z: 6, dim: 6, J: [0, 0, JN42] },
  ];

  const SOLIDS = [
    box(-6, 33, 0, GROUND), box(36, 56, 0, GROUND), box(80, 90, 0, GROUND),
    box(120, 138, 0, GROUND),
    box(23, 25, GROUND, GROUND + 2, "crate"),
    box(138, LEVEL_W, 0, TOP, "tower"),
    box(LEVEL_W, LEVEL_W + 4, 0, 40, "tower"),
    box(-8, -6, 0, 40, "tower"),
    box(157.5, 158.5, TOP, TOP + 2, "post"),
  ];
  for (const m of MAGNETS) {
    if (m.kind === "sphere") continue;
    const [w, h] = m.kind === "cuboid" ? [m.dim[0], m.dim[2]] : m.dim;
    SOLIDS.push({ ...box(m.x - w / 2, m.x + w / 2, m.z - h / 2, m.z + h / 2, "magnet"), magnet: m });
  }
  const SPHERES = MAGNETS.filter((m) => m.kind === "sphere");

  const CHECKPOINTS = [
    { x: 38, z: GROUND }, { x: 84, z: GROUND }, { x: 123, z: GROUND },
  ];
  const WORD = "Magpylib";
  const START_X = 9;
  const LETTER_Z = GROUND + 0.75;
  const LETTER_X = [11.1, 11.93, 12.61, 13.31, 13.97, 14.44, 14.8, 15.35]; /* set like the wordmark */
  /* where the magpie waits, always the right edge of something so its tail hangs free */
  const PERCHES = [
    { x: 82.7, z: 12, near: 12 },
    { x: 111.7, z: 8.5, near: 11 },
    { x: 158.2, z: TOP + 2, near: 14, last: true },
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
    ...arc(19, GROUND + 1, 3, 1.2, 0),
    ...arc(29, GROUND + 3, 4, 1.8, 2),
    ...arc(43, GROUND + 7.5, 3, 2, 0), /* over the cuboid, only a flip gets there */
    ...arc(60, 8, 5, 4, 0), /* under the rail */
    ...arc(94, 9, 3, 8, 1.5), /* over the cylinders */
    ...arc(116, 12, 2, 2, 0),
    ...arc(130, 12.5, 3, 2.5, 0.5), /* above the sphere */
    ...arc(145, TOP + 1.2, 5, 2, 0),
  ].map((c) => ({ ...c, got: false }));

  const NUTS = () => [
    { x: 30, z: GROUND, min: 26, max: 32.3, dir: -1, dead: 0 },
    { x: 87, z: GROUND, min: 86, max: 89.3, dir: 1, dead: 0 },
    { x: 150, z: TOP, min: 144, max: 156.5, dir: -1, dead: 0 },
  ];

  /* ---------- the field the hero feels ---------- */

  const tmp = [0, 0, 0];
  function absB(x, z) {
    tmp[0] = tmp[1] = tmp[2] = 0;
    for (const m of MAGNETS) {
      const half = m.kind === "cuboid" ? m.dim[0] / 2 : 3;
      if (Math.abs(x - m.x) < REACH + half) root.MagField.addB(m, x, 0, z, tmp);
    }
    return Math.hypot(tmp[0], tmp[1], tmp[2]);
  }

  function fieldAccel(p, out) {
    const cx = p.x, cz = p.z + H / 2;
    const gx = (absB(cx + DE, cz) - absB(cx - DE, cz)) / (2 * DE);
    const gz = (absB(cx, cz + DE) - absB(cx, cz - DE)) / (2 * DE);
    let ax = p.pol * K * gx, az = p.pol * K * gz;
    const a = Math.hypot(ax, az);
    if (a > FIELD_MAX) {
      ax *= FIELD_MAX / a;
      az *= FIELD_MAX / a;
    }
    out.x = ax;
    out.z = az;
  }

  /* ---------- state ---------- */

  function create() {
    return {
      phase: "title", /* title, intro, play, won */
      t: 0,
      p: {
        x: START_X, z: GROUND, vx: 0, vz: 0, pol: 1, face: 1,
        grounded: false, coyote: 0, buffer: 0, cut: false,
        contact: null, stuck: false, angle: 0, walk: 0, land: 0, flipT: 0, safe: 0,
        a: { x: 0, z: 0 },
      },
      prev: { jump: false, flip: false },
      check: { x: START_X, z: GROUND },
      sensors: SENSORS(),
      got: 0,
      nuts: NUTS(),
      deaths: 0,
      bird: { x: 0, z: 0, mode: "logo", route: [], perch: -1, face: 1, flap: 0 },
      letters: [...WORD].map((ch, i) => ({
        ch, i, x: LETTER_X[i], z: LETTER_Z, px: 0, pz: 0, on: false,
      })),
      rope: [],
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
      { x: LETTER_X[0] - 0.5, z: GROUND + 3, v: 11 },
      { x: LETTER_X[7] + 0.6, z: GROUND + 2.4, v: 7 },
      { x: 34, z: 13.5, v: 24 },
      { ...PERCHES[0], v: 27, perch: 0 },
    ];
    s.events.push("squawk");
  }

  /* ---------- the hero ---------- */

  function hit(p, s) {
    return p.x + W / 2 > s.x0 && p.x - W / 2 < s.x1 && p.z + H > s.z0 && p.z < s.z1;
  }

  function touch(p, nx, nz, magnet) {
    if (magnet) p.contact = { nx, nz };
  }

  function move(p, dt) {
    p.contact = null;
    const wasGrounded = p.grounded;
    p.grounded = false;

    p.x += p.vx * dt;
    for (const s of SOLIDS) {
      if (!hit(p, s)) continue;
      if (p.vx > 0) {
        p.x = s.x0 - W / 2;
        touch(p, -1, 0, s.magnet);
      } else if (p.vx < 0) {
        p.x = s.x1 + W / 2;
        touch(p, 1, 0, s.magnet);
      }
      p.vx = 0;
    }

    p.z += p.vz * dt;
    for (const s of SOLIDS) {
      if (!hit(p, s)) continue;
      if (p.vz <= 0) {
        p.z = s.z1;
        p.grounded = true;
        touch(p, 0, 1, s.magnet);
      } else {
        p.z = s.z0 - H;
        touch(p, 0, -1, s.magnet);
      }
      p.vz = 0;
    }

    for (const m of SPHERES) {
      const r = m.dim / 2;
      const qx = Math.max(p.x - W / 2, Math.min(m.x, p.x + W / 2));
      const qz = Math.max(p.z, Math.min(m.z, p.z + H));
      const dx = qx - m.x, dz = qz - m.z, d = Math.hypot(dx, dz);
      if (d >= r) continue;
      const nx = d > 0 ? dx / d : 0, nz = d > 0 ? dz / d : 1;
      p.x += nx * (r - d);
      p.z += nz * (r - d);
      const vn = p.vx * nx + p.vz * nz;
      if (vn < 0) {
        p.vx -= vn * nx;
        p.vz -= vn * nz;
      }
      if (nz > 0.5) p.grounded = true;
      touch(p, nx, nz, true);
    }

    if (!p.grounded && wasGrounded && p.vz <= 0) p.coyote = 0.1;
  }

  function respawn(s) {
    const p = s.p;
    p.x = s.check.x;
    p.z = s.check.z;
    p.vx = p.vz = 0;
    p.pol = 1;
    p.angle = 0;
    p.contact = null;
    p.stuck = false;
    p.safe = 1.5;
    s.deaths++;
    s.events.push("hurt");
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

    /* pressed on to a magnet hard enough, the hero walks on it whichever way up */
    const c = p.contact;
    p.stuck = !!c && p.pol > 0 && -(p.a.x * c.nx + p.a.z * c.nz) > 0.3 * G;

    if (p.stuck) {
      const sign = c.nz < -0.01 ? -1 : 1;
      const tx = sign * c.nz, tz = -sign * c.nx;
      const ax = p.a.x, az = p.a.z - G;
      let vt = p.vx * tx + p.vz * tz;
      let vn = p.vx * c.nx + p.vz * c.nz;
      vn += (ax * c.nx + az * c.nz) * dt;
      vt += (ax * tx + az * tz) * dt;
      if (dir) vt += dir * STUCK_ACC * dt;
      else vt -= Math.sign(vt) * Math.min(Math.abs(vt), STUCK_FRIC * dt);
      vt = Math.max(-RUN, Math.min(RUN, vt));
      p.vx = vt * tx + vn * c.nx;
      p.vz = vt * tz + vn * c.nz;
    } else {
      const on = p.grounded;
      if (dir && (Math.abs(p.vx) < RUN || dir * p.vx < 0)) {
        p.vx += dir * (on ? RUN_ACC : AIR_ACC) * dt;
      } else if (!dir && on) {
        p.vx -= Math.sign(p.vx) * Math.min(Math.abs(p.vx), RUN_FRIC * dt);
      }
      p.vx += p.a.x * dt;
      p.vz += (p.a.z - G) * dt;
    }

    p.coyote = Math.max(0, p.coyote - dt);
    p.buffer = Math.max(0, p.buffer - dt);
    /* pushing, the field under the hero is as good as a floor to jump from */
    const cushion = p.pol < 0 && p.a.z > 0.6 * G && p.vz < 4;
    if (p.buffer > 0 && (p.grounded || p.coyote > 0 || p.stuck || cushion)) {
      const nx = p.stuck ? c.nx : 0, nz = p.stuck ? c.nz : 1;
      p.vx += nx * JUMP_V;
      p.vz = p.stuck ? p.vz + nz * JUMP_V : JUMP_V;
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
    move(p, dt);
    if (p.grounded && was < -14) {
      p.land = 0.16;
    }

    /* poles first: rotate towards whichever magnet the hero is stuck on */
    const target = p.stuck && p.contact ? Math.atan2(p.contact.nx, p.contact.nz) : 0;
    let da = target - p.angle;
    while (da > Math.PI) da -= 2 * Math.PI;
    while (da < -Math.PI) da += 2 * Math.PI;
    p.angle += da * Math.min(1, dt * 14);

    p.walk += Math.abs(p.vx) * dt * 2.6;
    p.land = Math.max(0, p.land - dt);
    p.flipT = Math.max(0, p.flipT - dt);
    p.safe = Math.max(0, p.safe - dt);

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
      if (!sensor.got && Math.abs(sensor.x - p.x) < W / 2 + 0.3
          && sensor.z > p.z - 0.3 && sensor.z < p.z + H + 0.3) {
        sensor.got = true;
        s.got++;
        s.events.push("sensor");
      }
    }

    for (const n of s.nuts) {
      if (n.dead) continue;
      if (Math.abs(n.x - p.x) < W / 2 + 0.4 && p.z < n.z + 0.9 && p.z + H > n.z) {
        if (p.vz < 0 && p.z > n.z + 0.25) {
          n.dead = 0.5;
          p.vz = 13;
          s.events.push("stomp");
        } else if (!p.safe) {
          respawn(s);
          return;
        }
      }
    }
  }

  /* ---------- the magpie and its rope of stolen letters ---------- */

  /* on a perch the tail hangs over the edge, in flight it trails behind */
  function ropeAnchor(b) {
    return b.mode === "perched" ? [b.x + 0.55, b.z - 0.6] : [b.x - b.face * 0.45, b.z - 0.25];
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
          const tail = s.rope.pop();
          s.drop = { x: Math.min(tail.x, 156.8), z: tail.z, vx: -2.5, vz: 3, ch: tail.ch, rest: false };
          b.route = [{ x: 182, z: 22, v: 26 }];
        } else {
          b.route = [
            { x: b.x + 4, z: b.z + 3, v: 22 },
            { ...PERCHES[b.perch + 1], v: 27, perch: b.perch + 1 },
          ];
        }
      }
    }

    /* the magpie collects M first, each later letter goes on nearest its tail */
    for (const l of s.letters) {
      if (!l.on && b.mode === "fly" && b.x > l.x - 0.2 && s.phase !== "title") {
        l.on = true;
        l.px = l.x;
        l.pz = l.z;
        s.rope.unshift(l);
        s.events.push("pluck");
      }
    }

    let [ax, az] = ropeAnchor(b);
    for (const l of s.rope) {
      const nx = l.x + (l.x - l.px) * 0.985, nz = l.z + (l.z - l.pz) * 0.985 - 16 * dt * dt;
      l.px = l.x;
      l.pz = l.z;
      l.x = nx;
      l.z = nz;
      const dx = l.x - ax, dz = l.z - az, d = Math.hypot(dx, dz) || 1, rest = 1.02;
      if (d > rest) {
        l.x = ax + (dx / d) * rest;
        l.z = az + (dz / d) * rest;
      }
      /* the rope drags along the ground rather than sinking into it */
      for (const q of SOLIDS) {
        if (!q.magnet && l.x > q.x0 && l.x < q.x1 && l.z < q.z1 + 0.45) l.z = q.z1 + 0.45;
      }
      ax = l.x;
      az = l.z;
    }

    if (s.drop && !s.drop.rest) {
      const m = s.drop;
      m.vz -= 26 * dt;
      m.x += m.vx * dt;
      m.z += m.vz * dt;
      if (m.z < TOP + 0.05 && m.x < LEVEL_W && m.x > 138) {
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
    for (const n of s.nuts) {
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
    if (m && Math.abs(m.x - p.x) < W / 2 + 0.5 && m.z < p.z + H + 0.5 && m.z + 1 > p.z) {
      s.phase = "won";
      s.events.push("win");
    }
  }

  root.MagrioSim = {
    create, start, step, absB, ropeAnchor, STEP, W, H, GROUND, TOP, LEVEL_W, K,
    MAGNETS, SOLIDS, CHECKPOINTS, PERCHES, START_X,
  };
})(globalThis);
