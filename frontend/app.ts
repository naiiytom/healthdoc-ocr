import { bold, yellow } from "https://deno.land/std@0.56.0/fmt/colors.ts";
import { Router, Application, send, Context } from "https://deno.land/x/oak/mod.ts";

const router = new Router();
router
  .get("/", (context) => {
    context.response.status = 200,
    context.response.body = "Hello"
  });
  
const PORT = 8080;
const app = new Application();
app.use(router.routes());
app.use(router.allowedMethods());

console.log(
  bold("Start listening on ") + yellow(`http://localhost:${PORT}`)
);
await app.listen({ port: PORT });
