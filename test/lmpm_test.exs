defmodule LmpmTest do
  use ExUnit.Case
  doctest Lmpm

  test "greets the world" do
    assert Lmpm.hello() == :world
  end
end
