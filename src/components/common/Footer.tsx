import Link from 'next/link';
import { Github, Linkedin, Twitter } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';

export default function Footer() {
  return (
    <footer className="w-full bg-card text-card-foreground">
      <div className="container mx-auto grid grid-cols-1 gap-12 px-4 py-12 md:grid-cols-3">
        <div className="flex flex-col gap-4">
          <h3 className="font-headline text-lg font-semibold">DataFolio</h3>
          <p className="text-muted-foreground">
            My personal portfolio for showcasing data science projects and skills.
          </p>
          <div className="flex gap-2">
            <Button variant="ghost" size="icon" asChild>
              <Link href="#" aria-label="GitHub">
                <Github className="h-5 w-5" />
              </Link>
            </Button>
            <Button variant="ghost" size="icon" asChild>
              <Link href="#" aria-label="LinkedIn">
                <Linkedin className="h-5 w-5" />
              </Link>
            </Button>
            <Button variant="ghost" size="icon" asChild>
              <Link href="#" aria-label="Twitter">
                <Twitter className="h-5 w-5" />
              </Link>
            </Button>
          </div>
        </div>
        <div>
          <h3 className="mb-4 font-headline text-lg font-semibold">Quick Links</h3>
          <ul className="space-y-2">
            <li><Link href="/" className="text-muted-foreground hover:text-foreground">Home</Link></li>
            <li><Link href="/blog" className="text-muted-foreground hover:text-foreground">Blog</Link></li>
            <li><Link href="/summarizer" className="text-muted-foreground hover:text-foreground">AI Summarizer</Link></li>
          </ul>
        </div>
        <div>
          <h3 className="mb-4 font-headline text-lg font-semibold">Contact Me</h3>
          <form className="space-y-4">
            <Input type="email" placeholder="Your Email" aria-label="Your Email" />
            <Textarea placeholder="Your Message" aria-label="Your Message" />
            <Button type="submit" className="w-full bg-accent text-accent-foreground hover:bg-accent/90">Send Message</Button>
          </form>
        </div>
      </div>
      <div className="border-t">
        <div className="container mx-auto flex items-center justify-between px-4 py-4">
          <p className="text-sm text-muted-foreground">&copy; {new Date().getFullYear()} DataFolio. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
}
