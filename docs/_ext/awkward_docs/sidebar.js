// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
// Insert the shared sidebar tree in place of this script and mark the entry of
// the current page. awkward_docs.shared_sidebar.mark does the same in Python.
(function (tree) {
  "use strict";
  var script = document.currentScript;
  var page = script.getAttribute("data-page");
  var notRelative = /^([A-Za-z][A-Za-z0-9+.-]*:|\/|#)/;

  // sphinx.util.osutil.relative_uri
  function relativeUri(base, to) {
    var b = base.split("#")[0].split("/");
    var t = to.split("#")[0].split("/");
    while (b.length > 1 && t.length > 1 && b[0] === t[0]) {
      b.shift();
      t.shift();
    }
    if (b.join("/") === t.join("/") && b.length === t.length) {
      return "";
    }
    if (b.length === 1 && t.length === 1 && t[0] === "") {
      return "./";
    }
    return "../".repeat(b.length - 1) + t.join("/");
  }

  function markCurrent(anchor, holder) {
    anchor.setAttribute("aria-current", "page");
    anchor.className = ["current"].concat(anchor.className.split(/\s+/).filter(Boolean)).join(" ");
    for (var node = anchor.parentElement; node !== holder; node = node.parentElement) {
      var classes = node.className.split(/\s+/).filter(function (c) {
        return c && c !== "current" && c !== "active";
      });
      if (node.tagName === "LI") {
        node.className = classes.slice(0, 1).concat(["current", "active"], classes.slice(1)).join(" ");
        for (var i = 0; i < node.children.length; i++) {
          if (node.children[i].tagName === "DETAILS") {
            node.children[i].setAttribute("open", "open");
          }
        }
      } else if (node.tagName === "UL") {
        node.className = ["current"].concat(classes).join(" ");
      }
    }
  }

  var holder = document.createElement("div");
  holder.innerHTML = tree;
  var anchors = holder.querySelectorAll("a[href]");
  for (var i = 0; i < anchors.length; i++) {
    var anchor = anchors[i];
    var href = anchor.getAttribute("href");
    if (notRelative.test(href)) {
      continue;
    }
    var hash = href.indexOf("#");
    var path = hash < 0 ? href : href.slice(0, hash);
    var fragment = hash < 0 ? "" : href.slice(hash);
    if (path === page) {
      anchor.setAttribute("href", fragment || "#");
      markCurrent(anchor, holder);
    } else {
      anchor.setAttribute("href", relativeUri(page, path) + fragment);
    }
  }
  script.replaceWith.apply(script, Array.prototype.slice.call(holder.childNodes));
})(__AWKWARD_SIDEBAR_TREE__);
