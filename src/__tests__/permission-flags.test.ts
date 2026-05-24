import { describe, expect, it } from "bun:test";

const { permissionFlags } = await import("../index.js");

describe("aider permissionFlags", () => {
  it("plan → --chat-mode ask (no auto-apply)", () => {
    expect(permissionFlags("plan")).toEqual(["--chat-mode", "ask"]);
  });

  it("fullAuto → --yes-always", () => {
    expect(permissionFlags("fullAuto")).toEqual(["--yes-always"]);
  });

  it("acceptEdits → no flag", () => {
    expect(permissionFlags("acceptEdits")).toEqual([]);
  });

  it("undefined / unknown → acceptEdits (no --yes-always)", () => {
    expect(permissionFlags(undefined)).toEqual([]);
    expect(permissionFlags("bogus" as never)).toEqual([]);
  });
});
