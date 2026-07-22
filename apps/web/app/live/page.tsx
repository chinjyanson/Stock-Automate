import { redirect } from "next/navigation";

/** Live controls moved into Settings; keep the old path working. */
export default function LiveRedirect() {
  redirect("/settings");
}
