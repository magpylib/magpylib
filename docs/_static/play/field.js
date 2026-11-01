/*
 * Magpylib's magnet formulas, ported to plain JavaScript for the game.
 *
 * Scalar versions of magnet_cuboid_Bfield (field_BH_cuboid.py),
 * magnet_cylinder_axial_Bfield (field_BH_cylinder.py), the sphere branch of
 * _BHJM_magnet_sphere (field_BH_sphere.py), current_circle_Hfield
 * (field_BH_circle.py), _cel_scalar and _cel_iter_scalarvector (special_cel.py).
 * The variable names follow those files so the two can be read side by side.
 *
 * Lengths may be in any unit, B comes out in the unit of the polarization. A
 * magnet is { kind, x, z, dim, J } with its centre at (x, 0, z) and no rotation:
 *   cuboid    dim = [a, b, c]   J = [Jx, Jy, Jz]
 *   cylinder  dim = [d, h]      J = [0, 0, Jz]  (axial only)
 *   sphere    dim = d           J = [Jx, Jy, Jz]
 *   coil      dim = d, h, n, I  n loops of diameter d stacked over height h, I in
 *                               ampere, lengths in centimetres so that B is in tesla
 */
(function (root) {
  "use strict";

  const CEL_ERRORTOL = 1e-8;

  function cel(kc, p, c, s) {
    if (p > 0) {
      p = Math.sqrt(p);
      s /= p;
    } else {
      let f = kc * kc;
      let q = 1 - f;
      const g = 1 - p;
      f -= p;
      q *= s - c * p;
      p = Math.sqrt(f / g);
      c = (c - s) / g;
      s = -q / (g * g * p) + c * p;
    }
    let em = 1;
    let qc = Math.abs(kc);
    let kk = qc;
    for (;;) {
      let g = kk / p;
      const cNext = c + s / p;
      s = 2 * (s + c * g);
      c = cNext;
      p += g;
      g = em;
      em += qc;
      if (Math.abs(g - qc) <= g * CEL_ERRORTOL) break;
      qc = 2 * Math.sqrt(kk);
      kk = em * qc;
    }
    return ((Math.PI / 2) * (s + c * em)) / (em * (em + p));
  }

  function cuboid(x, y, z, dim, J, out) {
    const a = dim[0] / 2, b = dim[1] / 2, c = dim[2] / 2;
    /* evaluate in one octant only, the sign flips below map the result back */
    const sx = x < 0 ? -1 : 1, sy = y > 0 ? -1 : 1, sz = z > 0 ? -1 : 1;
    x *= sx; y *= sy; z *= sz;

    const xma = x - a, xpa = x + a, ymb = y - b, ypb = y + b, zmc = z - c, zpc = z + c;
    const xma2 = xma * xma, xpa2 = xpa * xpa, ymb2 = ymb * ymb;
    const ypb2 = ypb * ypb, zmc2 = zmc * zmc, zpc2 = zpc * zpc;

    const mmm = Math.sqrt(xma2 + ymb2 + zmc2), pmp = Math.sqrt(xpa2 + ymb2 + zpc2);
    const pmm = Math.sqrt(xpa2 + ymb2 + zmc2), mmp = Math.sqrt(xma2 + ymb2 + zpc2);
    const mpm = Math.sqrt(xma2 + ypb2 + zmc2), ppp = Math.sqrt(xpa2 + ypb2 + zpc2);
    const ppm = Math.sqrt(xpa2 + ypb2 + zmc2), mpp = Math.sqrt(xma2 + ypb2 + zpc2);

    const log = Math.log, at = Math.atan2;
    const ff2x = log((xma + mmm) * (xpa + ppm) * (xpa + pmp) * (xma + mpp))
      - log((xpa + pmm) * (xma + mpm) * (xma + mmp) * (xpa + ppp));
    const ff2y = log((-ymb + mmm) * (-ypb + ppm) * (-ymb + pmp) * (-ypb + mpp))
      - log((-ymb + pmm) * (-ypb + mpm) * (ymb - mmp) * (ypb - ppp));
    const ff2z = log((-zmc + mmm) * (-zmc + ppm) * (-zpc + pmp) * (-zpc + mpp))
      - log((-zmc + pmm) * (zmc - mpm) * (-zpc + mmp) * (zpc - ppp));

    const ff1x = at(ymb * zmc, xma * mmm) - at(ymb * zmc, xpa * pmm)
      - at(ypb * zmc, xma * mpm) + at(ypb * zmc, xpa * ppm)
      - at(ymb * zpc, xma * mmp) + at(ymb * zpc, xpa * pmp)
      + at(ypb * zpc, xma * mpp) - at(ypb * zpc, xpa * ppp);
    const ff1y = at(xma * zmc, ymb * mmm) - at(xpa * zmc, ymb * pmm)
      - at(xma * zmc, ypb * mpm) + at(xpa * zmc, ypb * ppm)
      - at(xma * zpc, ymb * mmp) + at(xpa * zpc, ymb * pmp)
      + at(xma * zpc, ypb * mpp) - at(xpa * zpc, ypb * ppp);
    const ff1z = at(xma * ymb, zmc * mmm) - at(xpa * ymb, zmc * pmm)
      - at(xma * ypb, zmc * mpm) + at(xpa * ypb, zmc * ppm)
      - at(xma * ymb, zpc * mmp) + at(xpa * ymb, zpc * pmp)
      + at(xma * ypb, zpc * mpp) - at(xpa * ypb, zpc * ppp);

    const qxy = sx * sy, qxz = sx * sz, qyz = sy * sz, k = 1 / (4 * Math.PI);
    out[0] += k * (J[0] * ff1x + J[1] * ff2z * qxy + J[2] * ff2y * qxz);
    out[1] += k * (J[0] * ff2z * qxy + J[1] * ff1y - J[2] * ff2x * qyz);
    out[2] += k * (J[0] * ff2y * qxz - J[1] * ff2x * qyz + J[2] * ff1z);
  }

  function cylinderAxial(x, y, z, dim, J, out) {
    const r0 = dim[0] / 2, z00 = dim[1] / 2;
    const rho = Math.sqrt(x * x + y * y);
    const r = rho / r0, zz = z / r0, z0 = z00 / r0;

    const zph = zz + z0, zmh = zz - z0, dpr = 1 + r, dmr = 1 - r;
    const sq0 = Math.sqrt(zmh * zmh + dpr * dpr), sq1 = Math.sqrt(zph * zph + dpr * dpr);
    const k1 = Math.sqrt((zph * zph + dmr * dmr) / (zph * zph + dpr * dpr));
    const k0 = Math.sqrt((zmh * zmh + dmr * dmr) / (zmh * zmh + dpr * dpr));
    if (k0 === 0 || k1 === 0) return; /* on the rim, where magpylib returns 0 as well */
    const gamma = dmr / dpr, g2 = gamma * gamma;

    const Br = (cel(k1, 1, 1, -1) / sq1 - cel(k0, 1, 1, -1) / sq0) / Math.PI;
    const Bz = ((1 / dpr) * ((zph * cel(k1, g2, 1, gamma)) / sq1
      - (zmh * cel(k0, g2, 1, gamma)) / sq0)) / Math.PI;

    if (rho > 0) {
      out[0] += (J[2] * Br * x) / rho;
      out[1] += (J[2] * Br * y) / rho;
    }
    out[2] += J[2] * Bz;
  }

  function sphere(x, y, z, d, J, out) {
    const r2 = x * x + y * y + z * z, rs = d / 2;
    if (r2 <= rs * rs) {
      out[0] += (J[0] * 2) / 3;
      out[1] += (J[1] * 2) / 3;
      out[2] += (J[2] * 2) / 3;
      return;
    }
    const k = (rs * rs * rs) / (3 * r2 * r2 * Math.sqrt(r2));
    const dot = 3 * (J[0] * x + J[1] * y + J[2] * z);
    out[0] += k * (dot * x - J[0] * r2);
    out[1] += k * (dot * y - J[1] * r2);
    out[2] += k * (dot * z - J[2] * r2);
  }

  function celIter(qc, p, g, cc, ss, em, kk) {
    while (Math.abs(g - qc) > g * CEL_ERRORTOL) {
      qc = 2 * Math.sqrt(kk);
      kk = em * qc;
      g = kk / p;
      const ccNext = cc + ss / p;
      ss = 2 * (ss + cc * g);
      cc = ccNext;
      p += g;
      g = em;
      em += qc;
    }
    return ((Math.PI / 2) * (ss + cc * em)) / (em * (em + p));
  }

  const MU0_PER_CM = 4e-7 * Math.PI * 100; /* mu0 times A per cm, in tesla */

  function circle(x, y, z, d, i0, out) {
    const r0 = d / 2, rho = Math.sqrt(x * x + y * y);
    const r = rho / r0, zz = z / r0;
    if (Math.abs(r - 1) < 1e-15 && zz === 0) return; /* on the wire, 0 in magpylib too */
    const z2 = zz * zz, x0 = z2 + (r + 1) * (r + 1);
    const k2 = (4 * r) / x0, q2 = (z2 + (r - 1) * (r - 1)) / x0;
    const q = Math.sqrt(q2), p = 1 + q;
    const pf = (MU0_PER_CM * i0) / (4 * Math.PI * r0 * Math.sqrt(x0) * q2);

    let cc = (k2 * 4 * zz) / x0, ss = (2 * cc * q) / p;
    const Br = pf * celIter(q, p, 1, cc, ss, p, q);
    const k4 = k2 * k2;
    cc = k4 - (q2 + 1) * (4 / x0);
    ss = 2 * q * (k4 / p - (4 / x0) * p);
    const Bz = -pf * celIter(q, p, 1, cc, ss, p, q);

    if (rho > 0) {
      out[0] += (Br * x) / rho;
      out[1] += (Br * y) / rho;
    }
    out[2] += Bz;
  }

  /* Adds one magnet's field at the world point (x, y, z) to out. */
  function addB(m, x, y, z, out) {
    if (m.kind === "coil") {
      for (let i = 0; i < m.n; i++) {
        circle(x - m.x, y, z - m.z - m.h * ((i + 0.5) / m.n - 0.5), m.dim, m.I, out);
      }
    } else if (m.kind === "cuboid") cuboid(x - m.x, y, z - m.z, m.dim, m.J, out);
    else if (m.kind === "cylinder") cylinderAxial(x - m.x, y, z - m.z, m.dim, m.J, out);
    else sphere(x - m.x, y, z - m.z, m.dim, m.J, out);
  }

  root.MagField = { addB };
})(globalThis);
