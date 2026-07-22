import { redirect } from "next/navigation";

// The portfolio and dashboard were merged into a single account view. Kept as a
// redirect so existing links and bookmarks still resolve.
export default function PortfolioPage() {
  redirect("/dashboard");
}
